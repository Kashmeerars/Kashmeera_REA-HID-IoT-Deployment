#!/usr/bin/env python3
"""
rea_hid_live_v2.py — FINAL LIVE ENGINE (FULL HYBRID ADAPTATION)
Run with strong drift: python3 rea_hid_live_v2.py --once --strong_drift
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import joblib

# Add utils directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.page_hinkley import PageHinkley
from utils.replay_buffer import ReplayBuffer

# TensorFlow / TFLite
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
except Exception as e:
    print(f"[WARN] TensorFlow import failed: {e}")
    keras_load_model = None

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter as TFLiteInterpreter
    except Exception:
        TFLiteInterpreter = None


FEATURE_ORDER = [
    'L4_SRC_PORT', 'IN_BYTES', 'OUT_BYTES', 'FLOW_DURATION_MILLISECONDS',
    'PROTOCOL', 'TCP_FLAGS', 'DURATION_IN', 'DURATION_OUT',
    'MIN_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'CLIENT_TCP_FLAGS',
    'IAT_mean'
]


class ModelWrapper:
    def __init__(self, keras_model=None, tflite_interpreter=None, input_details=None, output_details=None):
        self.keras_model = keras_model
        self.tflite = tflite_interpreter
        self.input_details = input_details
        self.output_details = output_details
        if self.tflite:
            self.tflite.allocate_tensors()

    def predict(self, X):
        if self.tflite:
            self.tflite.set_tensor(self.input_details[0]['index'], X.astype(np.float32))
            self.tflite.invoke()
            mlp = self.tflite.get_tensor(self.output_details[0]['index']).reshape(-1)
            recon = self.tflite.get_tensor(self.output_details[1]['index'])
            return mlp, recon
        else:
            preds = self.keras_model.predict(X, verbose=0)
            return preds[0].reshape(-1), preds[1]


def load_model(model_path, tflite_path=None, prefer_tflite=False):
    if prefer_tflite and tflite_path and TFLiteInterpreter:
        interp = TFLiteInterpreter(model_path=tflite_path)
        return ModelWrapper(
            tflite_interpreter=interp,
            input_details=interp.get_input_details(),
            output_details=interp.get_output_details()
        )
    model = keras_load_model(model_path)
    return ModelWrapper(keras_model=model)


def process_batch(df, model, scaler, threshold, ph, rb, args):
    stats = {k: 0 for k in ["processed", "attack", "benign", "pseudo1", "pseudo0", "drift", "retrain_trigger", "retrain_done"]}
    pre_retrain_probs = []
    post_retrain_probs = []
    has_retrained = False

    X_raw = df[FEATURE_ORDER].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    # Original model predictions (always available for comparison)
    orig_mlp_probs, orig_recon = model.predict(X_scaled)
    orig_errors = np.mean(np.abs(X_scaled - orig_recon), axis=1)

    for i in range(len(df)):
        stats["processed"] += 1

        # Get prediction from retrained hybrid model (if exists)
        retrained_mlp_prob, retrained_recon = rb.predict(X_raw[i])

        # Fallback to original model if no retrained model yet
        final_mlp_prob = retrained_mlp_prob if retrained_mlp_prob is not None else orig_mlp_probs[i]
        final_recon = retrained_recon if retrained_recon is not None else orig_recon[i]

        # Compute current reconstruction error using the active model
        current_error = np.mean(np.abs(X_scaled[i] - final_recon))

        # Live print only when using retrained model
        if retrained_mlp_prob is not None:
            print(f"Flow {stats['processed']:5d}: original MLP={orig_mlp_probs[i]:.3f} → "
                  f"retrained MLP={final_mlp_prob:.3f}  (error: {current_error:.4f})")
            has_retrained = True

        # Collect for final summary
        pre_retrain_probs.append(orig_mlp_probs[i])
        post_retrain_probs.append(final_mlp_prob)

        # Final hybrid decision
        pred = 1 if (final_mlp_prob > 0.5 or current_error > threshold) else 0
        stats["attack" if pred else "benign"] += 1

        # Pseudo-labeling (used for buffer)
        pseudo = 0
        if final_mlp_prob >= args.pseudo_mlp_conf:
            pseudo = 1
        elif current_error > threshold * args.pseudo_vae_factor:
            pseudo = 1
        stats["pseudo1" if pseudo else "pseudo0"] += 1

        # Add to replay buffer
        flow = np.concatenate([X_raw[i], [current_error, pseudo]])
        rb.add(flow)

        # Drift detection
        if ph.update(current_error):
            stats["drift"] += 1
            print(f"DRIFT DETECTED at flow {stats['processed']}")
            if rb.should_retrain():
                stats["retrain_trigger"] += 1
                before = rb.retrain_count
                rb.retrain_hybrid()  # Trains both MLP and AE
                if rb.retrain_count > before:
                    stats["retrain_done"] += 1
            ph.reset()

    return stats, pre_retrain_probs, post_retrain_probs, has_retrained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zeek_dir', default="../zeek_logs/live_eval_v2")
    parser.add_argument('--model_path', default="../models_v2/rea_hid_final.keras")
    parser.add_argument('--tflite_path', default="../models_v2/rea_hid_final_float16.tflite")
    parser.add_argument('--scaler_path', default="../models_v2/rea_hid_scaler.pkl")
    parser.add_argument('--threshold_path', default="../models_v2/rea_hid_threshold.npy")
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--prefer_tflite', action='store_true')
    parser.add_argument('--strong_drift', action='store_true')
    parser.add_argument('--ph_lambda', type=float, default=15.0)
    parser.add_argument('--ph_delta', type=float, default=0.002)
    parser.add_argument('--pseudo_mlp_conf', type=float, default=0.90)
    parser.add_argument('--pseudo_vae_factor', type=float, default=2.0)
    parser.add_argument('--buffer_vae_high_factor', type=float, default=1.2)
    args = parser.parse_args()

    scaler = joblib.load(args.scaler_path)
    threshold = float(np.load(args.threshold_path))
    model = load_model(args.model_path, args.tflite_path, args.prefer_tflite)

    vae_high = threshold * args.buffer_vae_high_factor
    ph = PageHinkley(delta=args.ph_delta, lambd=args.ph_lambda, min_instances=50)
    rb = ReplayBuffer(
        capacity=5000,
        min_samples=300,
        min_positive_fraction=0.005,
        vae_high_threshold=vae_high,
        scaler=scaler
    )

    csv_file = "live_drift_strong.csv" if args.strong_drift else "live_drift_v2_13feat.csv"
    path = os.path.join(args.zeek_dir, csv_file)

    print(f"\nREA-HID LIVE — Processing: {csv_file}")
    print(f"Threshold: {threshold:.6f} | PH λ={args.ph_lambda} δ={args.ph_delta}")

    df = pd.read_csv(path)
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    total, pre_probs, post_probs, has_retrained = process_batch(df, model, scaler, threshold, ph, rb, args)

    print("\n" + "="*60)
    print("FINAL RESULT")
    print(f"Flows processed   : {total['processed']}")
    print(f"Attack predicted  : {total['attack']} ({total['attack']/total['processed']*100:.2f}%)")
    print(f"Drift detections  : {total['drift']}")
    print(f"Retrain executed  : {total['retrain_done']}")
    print("="*60)
    rb.summary()

    if has_retrained:
        pre_mean = np.mean(pre_probs)
        post_mean = np.mean(post_probs)
        pre_extreme = np.sum(np.logical_or(np.array(pre_probs) < 0.01, np.array(pre_probs) > 0.99)) / len(pre_probs)
        post_extreme = np.sum(np.logical_or(np.array(post_probs) < 0.01, np.array(post_probs) > 0.99)) / len(post_probs)
        print("\n=== ADAPTATION SUMMARY ===")
        print("Before retraining:")
        print(f"  • Avg MLP confidence : {pre_mean:.3f} | Extreme scores: {pre_extreme*100:.1f}%")
        print("After full hybrid retraining:")
        print(f"  • Avg MLP confidence : {post_mean:.3f} | Extreme scores: {post_extreme*100:.1f}%")
        print(f"  • Overconfidence reduced by: {(pre_extreme - post_extreme)*100:.1f}%")
        print("  • Both MLP and AutoEncoder now adapted to new timing distribution")
        print("==========================")
    else:
        print("\nNo retraining occurred — no summary available.")


if __name__ == "__main__":
    main()