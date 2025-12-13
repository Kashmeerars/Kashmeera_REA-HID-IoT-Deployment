#!/usr/bin/env python3


import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def load_interpreter(path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        from tensorflow.lite import Interpreter

    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


def analyze_model(interp):
    inp = interp.get_input_details()[0]
    outs = interp.get_output_details()

    clf_idx = None
    recon_idx = None

    for o in outs:
        shape = o["shape"]
        if len(shape) == 2 and shape[1] == 1:
            clf_idx = o["index"]
        elif len(shape) == 2 and shape[1] == 13:
            recon_idx = o["index"]

    # Safe fallback
    if clf_idx is None:
        clf_idx = outs[0]["index"]
    if recon_idx is None:
        recon_idx = outs[1]["index"]

    return inp, clf_idx, recon_idx



def predict_single(interp, inp_detail, clf_idx, recon_idx, x_scaled):
    x = x_scaled.astype(np.float32).reshape(1, 13)
    interp.set_tensor(inp_detail["index"], x)
    interp.invoke()

    mlp_prob = float(interp.get_tensor(clf_idx)[0][0])
    recon = interp.get_tensor(recon_idx)[0]

    err = float(np.mean(np.abs(x[0] - recon)))
    return mlp_prob, err



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", default="rea_hid_final_float16.tflite")
    parser.add_argument("--scaler", default="rea_hid_scaler.pkl")
    parser.add_argument("--threshold", default="rea_hid_threshold.npy")
    parser.add_argument("--csv", default="demo_short.csv")
    parser.add_argument("--show_every", type=int, default=500)
    args = parser.parse_args()

    print("\n=== REA-HID RPi5 Inference + Evaluation ===\n")

 
    scaler = joblib.load(args.scaler)
    threshold = float(np.load(args.threshold))
    interp = load_interpreter(args.tflite)
    inp_detail, clf_idx, recon_idx = analyze_model(interp)

    print("Loaded model      :", args.tflite)
    print("Input shape       :", inp_detail["shape"])
    print("Classifier index  :", clf_idx)
    print("Reconstruction idx:", recon_idx)
    print("AE Threshold      :", threshold)
    print()

   
    df = pd.read_csv(args.csv)
    if "Label" in df.columns:
        y_true = df["Label"].values.astype(int)
    else:
        y_true = None
        print("[WARNING] CSV has no Label column → metrics will NOT be computed.\n")

    FEATURES = [
        "L4_SRC_PORT",
        "IN_BYTES",
        "OUT_BYTES",
        "FLOW_DURATION_MILLISECONDS",
        "PROTOCOL",
        "TCP_FLAGS",
        "DURATION_IN",
        "DURATION_OUT",
        "MIN_TTL",
        "LONGEST_FLOW_PKT",
        "SHORTEST_FLOW_PKT",
        "CLIENT_TCP_FLAGS",
        "IAT_mean",
    ]

    X_raw = df[FEATURES].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    N = len(df)
    print(f"Running inference on {N} flows...\n")

    y_pred = []
    latencies_ms = []

 
    for i in range(N):
        t0 = time.perf_counter()
        mlp_prob, err = predict_single(
            interp, inp_detail, clf_idx, recon_idx, X_scaled[i]
        )
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)

        pred = 1 if (mlp_prob > 0.5 or err > threshold) else 0
        y_pred.append(pred)

        if i < 10 or (i + 1) % args.show_every == 0 or i == N - 1:
            print(
                f"Flow {i+1:5d} | "
                f"MLP={mlp_prob:.3f} | "
                f"Err={err:.4f} | "
                f"Latency={latency_ms:.3f} ms | "
                f"→ {'ATTACK' if pred else 'BENIGN'}"
            )

    print("\n=== Inference Complete ===\n")

  
    latencies_ms = np.array(latencies_ms)
    print("=== Inference Latency Summary (RPi5) ===")
    print(f"Avg latency : {latencies_ms.mean():.4f} ms")
    print(f"P95 latency : {np.percentile(latencies_ms, 95):.4f} ms")
    print(f"Max latency : {latencies_ms.max():.4f} ms\n")


    if y_true is not None:
        print("=== Evaluation Metrics (RPi5) ===\n")

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}\n")

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))
    else:
        print("[INFO] No labels → skipping metric computation.")

    print("\n=== DONE — REA-HID RPi5 Inference & Latency Verified ===\n")


if __name__ == "__main__":
    main()
