import os, sys, json
from pathlib import Path
import numpy as np
 
sys.path.append(os.path.dirname(__file__))
from data import load_dataset
import joblib
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
)
from sklearn.inspection import permutation_importance
 
def evaluate(model_path: str, seed: int = 42):
    X, y, target_names = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
 
    bundle = joblib.load(model_path)
    model = bundle["pipeline"]
    meta = bundle.get("meta", {})
 
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
 
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "n_test": int(len(y_test)),
    }
 
    r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed)
    importances = sorted(
        [(X.columns[i], float(r.importances_mean[i])) for i in range(len(X.columns))],
        key=lambda x: abs(x[1]), reverse=True
    )[:10]
 
    report = {
        "meta": meta,
        "metrics": metrics,
        "top_permutation_importances": importances
    }
    print(json.dumps(report, indent=2))
    Path("reports").mkdir(exist_ok=True, parents=True)
    Path("reports/last_eval.json").write_text(json.dumps(report, indent=2))
    return report
 
if __name__ == "__main__":
    evaluate(os.path.join("app", "model.joblib"))