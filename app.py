"""Step 3 - Flask app: upload transactions.csv -> neural network scores it -> HTML report."""
import json, os, uuid
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow import keras
from flask import (Flask, render_template, request, send_file, redirect,
                   url_for, flash, abort)
from features import build_features, FEATURE_NAMES

app = Flask(__name__)
app.secret_key = "change-me"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.path.exists("model/gst_nn.keras"):
    raise SystemExit("Model not found. Run:  python generate_dataset.py  then  python train_model.py")
MODEL = keras.models.load_model("model/gst_nn.keras")
SCALER = joblib.load("model/scaler.pkl")
META = json.load(open("model/meta.json"))
THRESHOLD = 0.5


def reasons(row):
    """Plain-English indicators for a scored transaction (descriptive, from the input features)."""
    out = []
    if row.itc_ratio > 1.3:
        out.append(f"ITC claimed is {row.itc_ratio:.1f}x the tax payable")
    if row.near_threshold:
        out.append("Amount just below an approval threshold")
    if row.pair_txn_count >= 5 and row.reverse_pair_exists:
        out.append(f"Two-way trading with same party ({int(row.pair_txn_count)} invoices)")
    elif row.reverse_pair_exists:
        out.append("Buyer also sells back to this seller")
    if row.seller_age_days < 180 and row.log_seller_volume > np.log1p(1_000_000):
        out.append(f"New seller ({int(row.seller_age_days)} days) with high volume")
    if row.seller_unique_buyers <= 3 and row.log_seller_volume > np.log1p(2_000_000):
        out.append("Large volume concentrated on very few buyers")
    return out or ["Pattern flagged by neural network"]


def level(p):
    return "High" if p >= 0.8 else "Medium" if p >= THRESHOLD else "Low"


def analyze(df):
    feats = build_features(df)
    probs = MODEL.predict(SCALER.transform(feats[FEATURE_NAMES]), verbose=0).ravel()
    res = df.copy()
    res["fraud_probability"] = probs.round(4)
    res["risk_level"] = [level(p) for p in probs]
    res["flagged"] = (probs >= THRESHOLD).astype(int)
    res["indicators"] = [" | ".join(reasons(r)) for r in feats.itertuples()]
    return res, feats


@app.route("/")
def index():
    return render_template("index.html", meta=META)


@app.route("/sample")
def sample():
    return send_file("data/transactions.csv", as_attachment=True)


@app.route("/analyze", methods=["POST"])
def do_analyze():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        flash("Please choose a .csv file.")
        return redirect(url_for("index"))
    try:
        df = pd.read_csv(f)
        res, feats = analyze(df)
    except Exception as e:
        flash(f"Could not process file: {e}")
        return redirect(url_for("index"))

    rid = uuid.uuid4().hex[:12]
    res.to_csv(os.path.join(UPLOAD_DIR, f"{rid}.csv"), index=False)

    amt = pd.to_numeric(res["amount"], errors="coerce").fillna(0)
    fl = res[res.flagged == 1]
    itc_risk = 0.0
    if "itc_amount" in res.columns:
        itc_risk = float(pd.to_numeric(fl["itc_amount"], errors="coerce").fillna(0).sum())

    summary = {
        "total": len(res), "flagged": int(res.flagged.sum()),
        "flag_rate": round(100 * res.flagged.mean(), 1),
        "high": int((res.risk_level == "High").sum()),
        "medium": int((res.risk_level == "Medium").sum()),
        "low": int((res.risk_level == "Low").sum()),
        "flagged_amount": float(amt[res.flagged == 1].sum()),
        "total_amount": float(amt.sum()), "itc_at_risk": itc_risk,
    }

    id_col = "invoice_id" if "invoice_id" in res.columns else None
    top = res.sort_values("fraud_probability", ascending=False).head(25)
    top_rows = [dict(
        id=(r[id_col] if id_col else f"row {i}"), seller=r["seller_gstin"], buyer=r["buyer_gstin"],
        amount=float(r["amount"]), prob=float(r["fraud_probability"]),
        pct=round(float(r["fraud_probability"]) * 100, 1), level=r["risk_level"],
        indicators=r["indicators"].split(" | "),
    ) for i, r in top.iterrows()]

    sellers = (fl.groupby("seller_gstin")
                 .agg(flagged=("flagged", "sum"), avg_prob=("fraud_probability", "mean"),
                      amount=("amount", "sum"))
                 .sort_values(["flagged", "amount"], ascending=False).head(10).reset_index())
    tot = res.groupby("seller_gstin").size()
    seller_rows = [dict(gstin=r.seller_gstin, flagged=int(r.flagged), total=int(tot[r.seller_gstin]),
                        avg=round(float(r.avg_prob) * 100, 1), amount=float(r.amount))
                   for r in sellers.itertuples()]

    # ring-like pairs: flagged pairs that trade both ways
    pair_rows = []
    if len(fl):
        pc = fl.groupby(["seller_gstin", "buyer_gstin"]).agg(n=("amount", "size"), v=("amount", "sum")).reset_index()
        seen = set(zip(pc.seller_gstin, pc.buyer_gstin))
        for r in pc.sort_values("n", ascending=False).itertuples():
            if (r.buyer_gstin, r.seller_gstin) in seen and r.seller_gstin < r.buyer_gstin:
                pair_rows.append(dict(a=r.seller_gstin, b=r.buyer_gstin, n=int(r.n), v=float(r.v)))
        pair_rows = pair_rows[:8]

    # optional self-check if the uploaded file happens to contain labels
    check = None
    for c in ("is_fraud_label", "is_fraud", "label"):
        if c in res.columns:
            y = pd.to_numeric(res[c], errors="coerce").fillna(0).astype(int)
            tp = int(((res.flagged == 1) & (y == 1)).sum()); fp = int(((res.flagged == 1) & (y == 0)).sum())
            fn = int(((res.flagged == 0) & (y == 1)).sum()); tn = int(((res.flagged == 0) & (y == 0)).sum())
            check = dict(tp=tp, fp=fp, fn=fn, tn=tn, acc=round(100 * (tp + tn) / len(res), 2),
                         prec=round(100 * tp / max(tp + fp, 1), 1), rec=round(100 * tp / max(tp + fn, 1), 1))
            break

    return render_template("report.html", s=summary, top=top_rows, sellers=seller_rows,
                           pairs=pair_rows, meta=META, check=check, rid=rid,
                           filename=f.filename, now=datetime.now().strftime("%d %b %Y, %H:%M"))


@app.route("/download/<rid>")
def download(rid):
    if not rid.isalnum():
        abort(404)
    p = os.path.join(UPLOAD_DIR, f"{rid}.csv")
    if not os.path.exists(p):
        abort(404)
    df = pd.read_csv(p)
    out = os.path.join(UPLOAD_DIR, f"{rid}_flagged.csv")
    df[df.flagged == 1].sort_values("fraud_probability", ascending=False).to_csv(out, index=False)
    return send_file(out, as_attachment=True, download_name="flagged_transactions.csv")


@app.template_filter("inr")
def inr(v):
    return "₹{:,.0f}".format(v)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
