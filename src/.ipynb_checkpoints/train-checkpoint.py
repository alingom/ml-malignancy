import argparse, os, sys, json
from pathlib import Path
import numpy as np
 
sys.path.append(os.path.dirname(__file__))
from data import load_dataset
from models import get_model
 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import joblib
 
def train_and_save(model_name: str, out_path: str, seed: int = 42):
    X, y, target_names = load_dataset()
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
 
    model = get_model(model_name, random_state=seed)
 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"[CV] ROC-AUC: mean={scores.mean():.4f} ± {scores.std():.4f}")
 
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)
    print(f"[TEST] ROC-AUC: {test_auc:.4f}")
 
    meta = {
        "model_name": model_name,
        "seed": seed,
        "cv_roc_auc_mean": float(scores.mean()),
        "cv_roc_auc_std": float(scores.std()),
        "test_roc_auc": float(test_auc),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "features": list(X.columns),
        "target_names": target_names,
    }
 
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "meta": meta}, out)
    (out.parent / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] Saved model to {out}")
    return meta
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gb", help="lr | rf | gb")
    ap.add_argument("--out", default=os.path.join("app", "model.joblib"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    meta = train_and_save(args.model, args.out, args.seed)
    print(json.dumps(meta, indent=2))