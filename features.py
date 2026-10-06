"""Feature engineering shared by training and the Flask app.
Uses ONLY columns available in a plain transactions.csv (no label needed)."""
import numpy as np
import pandas as pd

THRESHOLDS = np.array([100, 500, 1000, 5000, 10000, 50000, 100000, 500000])

FEATURE_NAMES = [
    "log_amount", "tax_rate", "itc_ratio", "itc_excess",
    "near_threshold", "round_amount",
    "pair_txn_count", "reverse_pair_exists",
    "seller_unique_buyers", "seller_txn_count",
    "amount_vs_seller_avg", "log_seller_volume", "seller_age_days",
]

REQUIRED_COLUMNS = ["seller_gstin", "buyer_gstin", "amount"]


def validate(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing required column(s): " + ", ".join(missing))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    validate(df)
    f = pd.DataFrame(index=df.index)
    amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0).clip(lower=0)
    tax = pd.to_numeric(df.get("tax_rate", 18), errors="coerce").fillna(18)
    if "itc_amount" in df.columns:
        itc = pd.to_numeric(df["itc_amount"], errors="coerce").fillna(0)
    else:
        itc = amt * tax / 100.0
    expected_tax = (amt * tax / 100.0).clip(lower=1)

    f["log_amount"] = np.log1p(amt)
    f["tax_rate"] = tax
    f["itc_ratio"] = itc / expected_tax                # ~1.0 when honest
    f["itc_excess"] = np.log1p((itc - expected_tax).clip(lower=0))

    a = amt.values[:, None]
    f["near_threshold"] = ((a >= THRESHOLDS * 0.95) & (a < THRESHOLDS)).any(axis=1).astype(float)
    f["round_amount"] = ((amt % 1000 == 0) & (amt > 0)).astype(float)

    s, b = df["seller_gstin"].astype(str), df["buyer_gstin"].astype(str)
    pairs = pd.DataFrame({"s": s, "b": b})
    f["pair_txn_count"] = pairs.groupby(["s", "b"])["s"].transform("count").astype(float)
    pair_set = set(zip(s, b))
    f["reverse_pair_exists"] = [float((bb, ss) in pair_set) for ss, bb in zip(s, b)]
    f["seller_unique_buyers"] = pairs.groupby("s")["b"].transform("nunique").astype(float)
    f["seller_txn_count"] = pairs.groupby("s")["s"].transform("count").astype(float)

    tmp = pd.DataFrame({"s": s, "amt": amt})
    avg = tmp.groupby("s")["amt"].transform("mean").clip(lower=1)
    f["amount_vs_seller_avg"] = amt / avg
    f["log_seller_volume"] = np.log1p(tmp.groupby("s")["amt"].transform("sum"))

    if "seller_age_days" in df.columns:
        f["seller_age_days"] = pd.to_numeric(df["seller_age_days"], errors="coerce").fillna(1000)
    else:
        f["seller_age_days"] = 1000.0

    return f[FEATURE_NAMES].replace([np.inf, -np.inf], 0).fillna(0)
