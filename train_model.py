"""Step 2 - Train a neural network (ONLY a neural network) on the generated dataset."""
import json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, confusion_matrix)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from features import build_features, FEATURE_NAMES

SEED = 42
keras.utils.set_random_seed(SEED)
os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/train_dataset.csv")
X = build_features(df)
y = df["is_fraud_label"].astype(int).values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
scaler = RobustScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

model = keras.Sequential([
    keras.Input(shape=(Xtr_s.shape[1],)),
    layers.Dense(64, activation="relu"), layers.Dropout(0.3),
    layers.Dense(32, activation="relu"), layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="auc")])

w = (len(ytr) - ytr.sum()) / ytr.sum()
model.fit(Xtr_s, ytr, epochs=80, batch_size=128, validation_split=0.15, verbose=2,
          class_weight={0: 1.0, 1: float(w)},
          callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                   restore_best_weights=True)])

proba = model.predict(Xte_s, verbose=0).ravel()
pred = (proba >= 0.5).astype(int)
p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
auc = float(roc_auc_score(yte, proba))

# permutation importance (drop in ROC-AUC on held-out data)
rng = np.random.default_rng(SEED)
imp = []
for j, name in enumerate(FEATURE_NAMES):
    drops = []
    for _ in range(3):
        Xp = Xte_s.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        drops.append(auc - roc_auc_score(yte, model.predict(Xp, verbose=0).ravel()))
    imp.append({"feature": name, "auc_drop": round(float(np.mean(drops)), 4)})
imp.sort(key=lambda d: d["auc_drop"], reverse=True)

model.save("model/gst_nn.keras")
joblib.dump(scaler, "model/scaler.pkl")
meta = {
    "architecture": "Dense 64 > Dropout > 32 > Dropout > 16 > Sigmoid (Keras)",
    "train_rows": int(len(ytr)), "test_rows": int(len(yte)),
    "precision": round(float(p), 4), "recall": round(float(r), 4),
    "f1": round(float(f1), 4), "roc_auc": round(auc, 4),
    "pr_auc": round(float(average_precision_score(yte, proba)), 4),
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "feature_importance": imp[:8],
}
json.dump(meta, open("model/meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
