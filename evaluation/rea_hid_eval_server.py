#!/usr/bin/env python3
"""
rea_hid_eval_server.py
Offline evaluation script for AI server.
- Does NOT retrain
- Does NOT use ReplayBuffer or Page–Hinkley
- Computes Accuracy, Precision, Recall, F1, and Confusion Matrix
- Uses the same TFLite or Keras model used by RPi5
- Allows direct comparison between AI server inference and Raspberry Pi inference
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# --- Try loading TFLite interpreter ---
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter as TFLiteInterpreter
    except:
        TFLiteInterpreter = None

# --- Try loading Keras ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None


FEATURE_ORDER = [
    'L4_SRC_PORT', 'IN_BYTES', 'OUT_BYTES', 'FLOW_DURATION_MILLISECONDS',
    'PROTOCOL', 'TCP_FLAGS', 'DURATION_IN', 'DURATION_OUT',
    'MIN_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'CLIENT_TCP_FLAGS',
    'IAT_mean'
]


# ------------------------------------------------------------
# MODEL WRAPPER (same behavior as Pi script)
# ------------------------------------------------------------
class EvalModel:
    def __init__(self, keras_model=None, tflite=None, inp=None, clf_idx=None, recon_idx=None):
        self.keras_model = keras_model
        self.tflite = tflite
        self.inp = inp
        self.clf_idx = clf_idx
        self.recon_idx = recon_idx

    def predict_single(self, x_scaled):
        """Predict using TFLite if available, else Keras."""
        if self.tflite:
            x = x_scaled.astype(np.float32).reshape(1, 13)
            self.tflite.set_tensor(self.inp['index'], x)
            self.tflite.invoke()

            prob = float(self.tflite.get_tensor(self.clf_idx)[0][0])
            recon = self.tflite.get_tensor(self.recon_idx)[0]

            err = float(np.mean(np.abs(x[0] - recon)))
            return prob, err

        # keras fallback
        prob, recon = self.keras_model.predict(x_scaled.reshape(1, 13), verbose=0)
        prob = float(prob[0][0])
        recon = recon[0]
        err = float(np.mean(np.abs(x_scaled - recon)))
        return prob, err


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
def load_eval_model(model_path, tflite_path, prefer_tflite=False):
    """Load TFLite model if available, else load Keras model."""
    if prefer_tflite and TFLiteInterpreter and os.path.exists(tflite_path):
        print("[INFO] Using TFLite model for evaluation.")
        interpreter = TFLiteInterpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]

        outs = interpreter.get_output_details()
        clf_idx = outs[0]['index']
        recon_idx = outs[1]['index']

        return EvalModel(keras_model=None,
                         tflite=interpreter,
                         inp=inp,
                         clf_idx=clf_idx,
                         recon_idx=recon_idx)

    else:
        print("[INFO] Using Keras model for evaluation.")
        model = keras_load_model(model_path)
        return EvalModel(keras_model=model)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../zeek_logs/live_eval_v2/demo_short.csv")
    parser.add_argument("--model_path", default="../models_v2/rea_hid_final.keras")
    parser.add_argument("--tflite_path", default="../models_v2/rea_hid_final_float16.tflite")
    parser.add_argument("--scaler_path", default="../models_v2/rea_hid_scaler.pkl")
    parser.add_argument("--threshold_path", default="../models_v2/rea_hid_threshold.npy")
    parser.add_argument("--prefer_tflite", action="store_true")
    args = parser.parse_args()

    print("\n=== REA-HID AI Server Evaluation (Offline) ===\n")

    # Load scaler + threshold
    scaler = joblib.load(args.scaler_path)
    threshold = float(np.load(args.threshold_path))
    print(f"Loaded threshold: {threshold:.6f}")

    # Load model
    model = load_eval_model(args.model_path, args.tflite_path, args.prefer_tflite)

    # Load CSV
    df = pd.read_csv(args.csv)

    if "Label" not in df.columns:
        print("[ERROR] CSV must contain 'Label' column for evaluation.")
        sys.exit(1)

    y_true = df["Label"].values.astype(int)
    X_raw = df[FEATURE_ORDER].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    y_pred = []
    mlp_list = []
    err_list = []

    print(f"Evaluating {len(df)} flows...\n")

    for i in range(len(df)):
        mlp, err = model.predict_single(X_scaled[i])
        mlp_list.append(mlp)
        err_list.append(err)

        pred = 1 if (mlp > 0.5 or err > threshold) else 0
        y_pred.append(pred)

        if i < 10 or (i + 1) % 500 == 0 or i == len(df) - 1:
            print(f"Flow {i+1:5d} | MLP={mlp:.3f} | Err={err:.4f} | → {'ATTACK' if pred else 'BENIGN'}")

    # Metrics
    print("\n=== FINAL METRICS (AI Server Offline) ===")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("\n=== DONE: AI Server Inference Complete ===\n")


if __name__ == "__main__":
    main()
