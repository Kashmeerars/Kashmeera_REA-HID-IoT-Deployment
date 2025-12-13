#!/usr/bin/env python3


import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter as TFLiteInterpreter
    except Exception:
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


class EvalModel:
    def __init__(self, keras_model=None, tflite=None, inp=None, clf_idx=None, recon_idx=None):
        self.keras_model = keras_model
        self.tflite = tflite
        self.inp = inp
        self.clf_idx = clf_idx
        self.recon_idx = recon_idx

    def predict_single(self, x_scaled):
        if self.tflite:
            x = x_scaled.astype(np.float32).reshape(1, 13)
            self.tflite.set_tensor(self.inp['index'], x)
            self.tflite.invoke()

            prob = float(self.tflite.get_tensor(self.clf_idx)[0][0])
            recon = self.tflite.get_tensor(self.recon_idx)[0]
            err = float(np.mean(np.abs(x[0] - recon)))
            return prob, err

        # Keras fallback
        prob, recon = self.keras_model.predict(x_scaled.reshape(1, 13), verbose=0)
        prob = float(prob[0][0])
        recon = recon[0]
        err = float(np.mean(np.abs(x_scaled - recon)))
        return prob, err



def load_eval_model(model_path, tflite_path, prefer_tflite=False):
    if prefer_tflite and TFLiteInterpreter and os.path.exists(tflite_path):
        print("[INFO] Using TFLite model for evaluation.")
        interpreter = TFLiteInterpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        outs = interpreter.get_output_details()
        clf_idx = outs[0]['index']
        recon_idx = outs[1]['index']

        return EvalModel(
            keras_model=None,
            tflite=interpreter,
            inp=inp,
            clf_idx=clf_idx,
            recon_idx=recon_idx,
        )
    else:
        print("[INFO] Using Keras model for evaluation.")
        model = keras_load_model(model_path)
        return EvalModel(keras_model=model)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tflite_path", required=True)
    parser.add_argument("--scaler_path", required=True)
    parser.add_argument("--threshold_path", required=True)
    parser.add_argument("--prefer_tflite", action="store_true")
    args = parser.parse_args()

    print("\n=== REA-HID AI Server Evaluation (Offline) ===\n")

    scaler = joblib.load(args.scaler_path)
    threshold = float(np.load(args.threshold_path))
    print(f"Loaded threshold: {threshold:.6f}")

    model = load_eval_model(args.model_path, args.tflite_path, args.prefer_tflite)

    df = pd.read_csv(args.csv)
    if "Label" not in df.columns:
        print("[ERROR] CSV must contain 'Label' column.")
        sys.exit(1)

    y_true = df["Label"].values.astype(int)
    X_raw = df[FEATURE_ORDER].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    y_pred = []
    latencies_ms = []

    print(f"Evaluating {len(df)} flows...\n")

    for i in range(len(df)):
        t0 = time.perf_counter()
        mlp, err = model.predict_single(X_scaled[i])
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

        pred = 1 if (mlp > 0.5 or err > threshold) else 0
        y_pred.append(pred)

        if i < 10 or (i + 1) % 500 == 0 or i == len(df) - 1:
            print(
                f"Flow {i+1:5d} | "
                f"MLP={mlp:.3f} | Err={err:.4f} | "
                f"Latency={latencies_ms[-1]:.3f} ms | "
                f"→ {'ATTACK' if pred else 'BENIGN'}"
            )

    latencies_ms = np.array(latencies_ms)

    print("\n=== Inference Latency Summary (AI Server) ===")
    print(f"Avg latency : {latencies_ms.mean():.4f} ms")
    print(f"P95 latency : {np.percentile(latencies_ms, 95):.4f} ms")
    print(f"Max latency : {latencies_ms.max():.4f} ms\n")

    print("=== FINAL METRICS (AI Server) ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, zero_division=0):.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("\n=== DONE — AI Server Inference & Latency Verified ===\n")


if __name__ == "__main__":
    main()
