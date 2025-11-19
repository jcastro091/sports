# src/train_trades.py
from pathlib import Path
import argparse, joblib
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from .preprocess_trades import clean_and_prepare_trades


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODEL_DIR = DATA / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main(csv_name="ConfirmedBets - ConfirmedTrades2.csv"):
    X, y, pre, num_cols, cat_cols = clean_and_prepare_trades(str(DATA / csv_name))

    n_samples = len(y)
    counts = Counter(y)
    print(f"[train_trades] Class counts: {dict(counts)} | n={n_samples}")
    min_class = min(counts.values()) if counts else 0

    # ── Split strategy ────────────────────────────────────────────────────────────
    use_strat = min_class >= 2
    if use_strat:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
    else:
        print("[warn] Minority class < 2; disabling stratify; using 80/20 split.")
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )

    # ── Models ───────────────────────────────────────────────────────────────────
    base_lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    rf = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )

    # Only calibrate when we have enough data per class for stable CV
    # (need >= 3 per class so that cv>=3 is valid; also require a modest total sample size)
    can_calibrate = (min_class >= 3) and (n_samples >= 30)
    if can_calibrate:
        cv_splits = min(5, min_class)  # with min_class>=3, this is 3..5
        lr = CalibratedClassifierCV(base_lr, method="isotonic", cv=cv_splits)
        print(f"[info] Using calibrated LR with cv={cv_splits}.")
    else:
        lr = base_lr
        reason = []
        if min_class < 3: reason.append(f"min_class={min_class} (<3)")
        if n_samples < 30: reason.append(f"n_samples={n_samples} (<30)")
        print(f"[info] Using base LR (no calibration): {', '.join(reason)}")

    # ── Train with safe fallback if calibration still complains ───────────────────
    try:
        lr.fit(Xtr, ytr)
    except ValueError as e:
        print(f"[fallback] Calibration failed: {e}")
        print("[fallback] Retrying with uncalibrated LogisticRegression.")
        lr = base_lr
        lr.fit(Xtr, ytr)

    rf.fit(Xtr, ytr)

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # ── Eval: Logistic Regression ────────────────────────────────────────────────
    print("\n🔍 Logistic Regression:")
    ypred = lr.predict(Xte)
    try:
        yproba = lr.predict_proba(Xte)[:, 1]
        try:
            auc = roc_auc_score(yte, yproba)
            print(f"AUC: {auc:.3f}")
        except Exception:
            pass
    except Exception:
        pass
    print(classification_report(yte, ypred, zero_division=0))
    print(confusion_matrix(yte, ypred))

    # ── Eval: Random Forest ─────────────────────────────────────────────────────
    print("\n🌲 Random Forest:")
    ypred_rf = rf.predict(Xte)
    print(classification_report(yte, ypred_rf, zero_division=0))
    print(confusion_matrix(yte, ypred_rf))

    # Save RF bundle first (swap later if LR wins)
    joblib.dump(
        {"model": rf, "pre": pre, "num": num_cols, "cat": cat_cols},
        str(MODEL_DIR / "baseline_trades.pkl"),
    )
    print(f"💾 Saved {MODEL_DIR / 'baseline_trades.pkl'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ConfirmedBets - ConfirmedTrades2.csv")
    args = parser.parse_args()
    main(args.dataset)
