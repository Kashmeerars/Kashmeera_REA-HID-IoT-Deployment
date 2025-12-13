#!/usr/bin/env python3


import os
import random
import json
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


SEED = 42
DATA_CSV = os.path.expanduser("~/projects/replay_emulation_ids/zeek_logs/synthetic_training_v2/synthetic_training_v2_13feat.csv")
OUT_DIR = os.path.expanduser("~/projects/replay_emulation_ids/models_v2")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_ORDER = [
    "L4_SRC_PORT", "IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS",
    "PROTOCOL", "TCP_FLAGS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "CLIENT_TCP_FLAGS",
    "IAT_mean"
]

MODEL_PATH = os.path.join(OUT_DIR, "rea_hid_final.keras")
SCALER_PATH = os.path.join(OUT_DIR, "rea_hid_scaler.pkl")
THRESH_PATH = os.path.join(OUT_DIR, "rea_hid_threshold.npy")
FEATURE_JSON = os.path.join(OUT_DIR, "feature_order.json")
TFLITE_PATH = os.path.join(OUT_DIR, "rea_hid_final_float16.tflite")

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

df = pd.read_csv(DATA_CSV)[FEATURE_ORDER + ["Label"]]
X = df[FEATURE_ORDER].astype(np.float32).values
y = df["Label"].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)


scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

joblib.dump(scaler, SCALER_PATH)
with open(FEATURE_JSON, "w") as f:
    json.dump(FEATURE_ORDER, f)


inp = Input(shape=(13,))
x = Dense(128, activation="relu")(inp)
x = Dense(64, activation="relu")(x)
clf = Dense(1, activation="sigmoid", name="classifier")(x)

# Reconstruction path
e = Dense(32, activation="relu")(x)
e = Dense(16, activation="relu")(e)
d = Dense(32, activation="relu")(e)
d = Dense(64, activation="relu")(d)
recon = Dense(13, activation=None, name="reconstruction")(d)

model = Model(inp, [clf, recon])
model.compile(
    optimizer="adam",
    loss={"classifier": "binary_crossentropy", "reconstruction": "mse"},
    loss_weights={"classifier": 1.0, "reconstruction": 0.1},
    metrics={"classifier": "accuracy"}
)


model.fit(
    X_train_s,
    {"classifier": y_train, "reconstruction": X_train_s},
    validation_data=(X_val_s, {"classifier": y_val, "reconstruction": X_val_s}),
    epochs=50,
    batch_size=64,
    verbose=2,
    callbacks=[EarlyStopping(monitor="val_classifier_accuracy", mode="max", patience=8, restore_best_weights=True)]
)

model.save(MODEL_PATH)


val_clf, val_recon = model.predict(X_val_s, verbose=0)
val_errors = np.mean(np.abs(X_val_s - val_recon), axis=1)
benign_errors = val_errors[y_val == 0]
threshold = np.percentile(benign_errors, 95)
np.save(THRESH_PATH, np.array([threshold]))


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)


test_clf, test_recon = model.predict(X_test_s, verbose=0)
test_mlp = (test_clf.ravel() > 0.5).astype(int)
test_vae = (np.mean(np.abs(X_test_s - test_recon), axis=1) > threshold).astype(int)
test_ens = np.logical_or(test_mlp, test_vae).astype(int)

print("\n" + "="*60)
print(f"F1 MLP-only : {f1_score(y_test, test_mlp):.4f}")
print(f"F1 VAE-only : {f1_score(y_test, test_vae):.4f}")
print(f"F1 Ensemble : {f1_score(y_test, test_ens):.4f}")
print(f"Threshold   : {threshold:.6f}")
print(f"All artifacts saved to: {OUT_DIR}")
print("="*60)
