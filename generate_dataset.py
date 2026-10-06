"""Step 1 - Generate a synthetic GST transactions dataset.

Outputs:
  data/train_dataset.csv   (with is_fraud_label)  -> used to train the NN
  data/transactions.csv    (no label)              -> sample file to upload in the web app
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
os.makedirs("data", exist_ok=True)


def gstin(prefix, i):
    return f"{prefix}{i:05d}1Z{i % 10}"


def make_dataset(n_normal=18000, n_fraud=2000, seed=42):
    rng = np.random.default_rng(seed)
    honest = [gstin("27AABC", i) for i in range(600)]
    honest_age = {g: int(rng.integers(400, 4000)) for g in honest}
    shells = [gstin("29SHELL", i) for i in range(40)]          # 8 rings of 5
    shell_age = {g: int(rng.integers(20, 170)) for g in shells}
    start = pd.Timestamp("2025-04-01")

    rows = []

    def add(seller, buyer, amount, tax, itc, age, fraud, kind):
        rows.append(dict(
            invoice_date=(start + pd.Timedelta(days=int(rng.integers(0, 365)))).date(),
            seller_gstin=seller, buyer_gstin=buyer,
            amount=round(float(amount), 2), tax_rate=int(tax),
            itc_amount=round(float(itc), 2), seller_age_days=age,
            is_fraud_label=fraud, fraud_type=kind))

    # ---- normal trade (log-normal amounts, honest ITC, a few round/near-threshold by chance)
    for _ in range(n_normal):
        s, b = rng.choice(honest, 2, replace=False)
        amt = float(np.exp(rng.normal(9.0, 1.4)))
        if rng.random() < 0.04:
            amt = float(rng.choice([1000, 2000, 5000, 10000, 25000]))        # innocent round amounts
        if rng.random() < 0.02:
            amt = float(rng.choice(THR) * rng.uniform(0.95, 0.999))          # innocent near-threshold
        tax = rng.choice([5, 12, 18, 28], p=[.15, .25, .45, .15])
        itc = amt * tax / 100 * rng.uniform(0.97, 1.03) if rng.random() < .9 else 0
        add(s, b, amt, tax, itc, honest_age[s], 0, "normal")

    # ---- fraud patterns
    kinds = rng.choice(["circular", "threshold", "itc_inflation", "shell_burst"],
                       n_fraud, p=[.35, .25, .25, .15])
    for k in kinds:
        tax = rng.choice([12, 18, 28])
        if k == "circular":                                   # ring A->B->C->D->E->A
            r = int(rng.integers(0, 8)) * 5
            i = int(rng.integers(0, 5))
            s, b = shells[r + i], shells[r + (i + 1) % 5]
            amt = rng.uniform(80_000, 900_000)
            add(s, b, amt, tax, amt * tax / 100, shell_age[s], 1, k)
        elif k == "threshold":                                # amounts hugging approval limits
            s, b = rng.choice(honest, 2, replace=False)
            amt = rng.choice(THR) * rng.uniform(0.95, 0.9995)
            add(s, b, amt, tax, amt * tax / 100, honest_age[s], 1, k)
        elif k == "itc_inflation":                            # ITC claimed >> tax actually paid
            s, b = rng.choice(honest, 2, replace=False)
            amt = float(np.exp(rng.normal(9.5, 1.2)))
            add(s, b, amt, tax, amt * tax / 100 * rng.uniform(1.6, 3.5), honest_age[s], 1, k)
        else:                                                 # brand-new shell, huge volume to few buyers
            s = rng.choice(shells)
            b = rng.choice(honest[:6])
            amt = rng.uniform(150_000, 1_200_000)
            add(s, b, amt, tax, amt * tax / 100 * rng.uniform(1.0, 1.4), shell_age[s], 1, k)

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "invoice_id", [f"INV{i + 1:06d}" for i in range(len(df))])
    return df


THR = np.array([100, 500, 1000, 5000, 10000, 50000, 100000, 500000])

if __name__ == "__main__":
    full = make_dataset()
    full.to_csv("data/train_dataset.csv", index=False)

    # Fresh, unseen file (different seed) with the label removed = what the user uploads
    test = make_dataset(n_normal=2700, n_fraud=300, seed=7)
    test.drop(columns=["is_fraud_label", "fraud_type"]).to_csv("data/transactions.csv", index=False)
    test.to_csv("data/transactions_with_answers.csv", index=False)   # only for checking results

    print(f"train_dataset.csv : {len(full)} rows, fraud rate {full.is_fraud_label.mean():.1%}")
    print(f"transactions.csv  : {len(test)} rows (no label, upload this in the app)")
