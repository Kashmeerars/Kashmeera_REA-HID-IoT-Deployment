# utils/replay_buffer.py
import numpy as np
import collections
import time
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler


class ReplayBuffer:
    

    def __init__(
        self,
        capacity=5000,
        retrain_ratio=0.95,
        retrain_interval=500,
        min_samples=500,
        scaler=None,
        min_positive_fraction=0.005,
        anomaly_rate_trigger=0.10,
        vae_high_threshold=None
    ):
        self.buffer = collections.deque(maxlen=capacity)
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.fitted = scaler is not None

        
        self.retrain_ratio = float(retrain_ratio)
        self.retrain_interval = int(retrain_interval)
        self.min_samples = int(min_samples)
        self.min_positive_fraction = float(min_positive_fraction)
        self.anomaly_rate_trigger = float(anomaly_rate_trigger)
        self.vae_high_threshold = vae_high_threshold

        
        self.retrain_count = 0
        self.mlp = None
        self.autoencoder = None
        self.hybrid_model = None  # final model used for inference

        
        self.artifact_dir = os.path.join(os.getcwd(), "retrain_artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)

   
    def add(self, flow):
        if len(flow) != 15:
            print(f"[ReplayBuffer] Warning: Bad flow length={len(flow)}, expected 15. Skipping.")
            return
        self.buffer.append(flow)

    
    def should_retrain(self):
        n = len(self.buffer)
        if n < self.min_samples:
            return False

        labels = np.array([int(f[-1]) for f in self.buffer])
        positive_fraction = labels.mean()
        benign_ratio = 1.0 - positive_fraction

        if self.vae_high_threshold is None:
            return False
        vae_errors = np.array([float(f[13]) for f in self.buffer])
        anomaly_fraction = np.mean(vae_errors > self.vae_high_threshold)

        imbalance_trigger = (benign_ratio > self.retrain_ratio) or (benign_ratio < (1 - self.retrain_ratio))
        periodic_trigger = (n % self.retrain_interval == 0)
        anomaly_trigger = (anomaly_fraction >= self.anomaly_rate_trigger)

        triggered = imbalance_trigger or periodic_trigger or anomaly_trigger

        if triggered and positive_fraction >= self.min_positive_fraction:
            print(f"[ReplayBuffer] Retrain triggered | Size={n} | PosFrac={positive_fraction:.4f} "
                  f"| BenignRatio={benign_ratio:.4f} | Anomaly={anomaly_fraction:.4f}")
            return True
        elif triggered:
            print(f"[ReplayBuffer] Retrain skipped — not enough pseudo-positive samples")
        return False

    
    def retrain_hybrid(self):
        
        n = len(self.buffer)
        if n < self.min_samples:
            print(f"[ReplayBuffer] Not enough samples ({n}) for retrain.")
            return

       
        X_raw = np.array([f[:-2] for f in self.buffer], dtype=np.float32)
        y = np.array([int(f[-1]) for f in self.buffer], dtype=np.int32)

        if not self.fitted:
            X_scaled = self.scaler.fit_transform(X_raw)
            self.fitted = True
            try:
                import joblib
                joblib.dump(self.scaler, os.path.join(self.artifact_dir, "replay_scaler.pkl"))
            except:
                pass
        else:
            X_scaled = self.scaler.transform(X_raw)

        
        pos_frac = y.mean()
        class_weight = None
        if 0 < pos_frac < 0.3:
            class_weight = {0: 1.0, 1: max(1.0, (1 - pos_frac) / max(pos_frac, 1e-6))}
            print(f"[ReplayBuffer] Using class_weight: {class_weight}")

        mlp = Sequential([
            Dense(32, activation='relu', input_shape=(13,)),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid', name='classifier')
        ])
        mlp.compile(optimizer='adam', loss='binary_crossentropy')
        mlp.fit(X_scaled, y, epochs=5, batch_size=64, verbose=0, class_weight=class_weight)

        
        benign_mask = (y == 0)
        if np.sum(benign_mask) < 100:
            print("[ReplayBuffer] Warning: Not enough pseudo-benign samples for AE training. Skipping AE update.")
            autoencoder = None
        else:
            X_benign = X_scaled[benign_mask]
            input_layer = Input(shape=(13,))
            e = Dense(16, activation='relu')(input_layer)
            e = Dense(8, activation='relu')(e)
            d = Dense(16, activation='relu')(e)
            decoded = Dense(13, activation=None)(d)
            autoencoder = Model(input_layer, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(X_benign, X_benign, epochs=12, batch_size=64, verbose=0)
            print(f"[ReplayBuffer] AutoEncoder retrained on {X_benign.shape[0]} pseudo-benign flows")
        
        input_layer = Input(shape=(13,), name='input')
        clf_out = mlp(input_layer)
        recon_out = autoencoder(input_layer) if autoencoder else input_layer  # fallback = identity
        hybrid = Model(inputs=input_layer, outputs=[clf_out, recon_out])

        self.hybrid_model = hybrid
        self.mlp = mlp
        self.autoencoder = autoencoder
        self.retrain_count += 1

        
        fname = os.path.join(self.artifact_dir, f"rea_hid_hybrid_retrain_{self.retrain_count}.keras")
        hybrid.save(fname)
        meta = {
            "samples": int(n),
            "positives": int(y.sum()),
            "positive_fraction": float(pos_frac),
            "benign_for_ae": int(np.sum(benign_mask)),
            "timestamp": int(time.time())
        }
        with open(fname + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[ReplayBuffer] Full HYBRID model (MLP + AE) #{self.retrain_count} saved → {os.path.basename(fname)}")

   
    def predict(self, raw_features_13):
        
        if self.hybrid_model is None:
            return None, None  # no model yet

        X_scaled = self.scaler.transform([raw_features_13])
        prob, recon = self.hybrid_model.predict(X_scaled, verbose=0)
        return float(prob[0][0]), recon[0]

   
    def summary(self):
        n = len(self.buffer)
        if n == 0:
            print("[ReplayBuffer] Buffer empty.")
            return
        labels = np.array([int(f[-1]) for f in self.buffer], dtype=np.int32)
        pos = labels.sum()
        benign_ratio = 1 - pos / n
        print("======== REPLAYBUFFER SUMMARY ========")
        print(f"Size: {n}")
        print(f"Positives: {pos}")
        print(f"BenignRatio: {benign_ratio:.3f}")
        print(f"Retrains: {self.retrain_count}")
        print(f"Latest hybrid model: {'Yes' if self.hybrid_model else 'No'}")
        print("======================================")
