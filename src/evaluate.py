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

import mlflow
from eval_utils import (
    plot_calibration_curve,
    delong_roc_test,
    mcnemar_test,
    subgroup_performance,
    equalized_odds_check,
)
 
def evaluate(model_path: str, seed: int = 42, compare_model_path: str = None, sensitive_column: str = None):
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
    # Fairness checks: subgroup performance and equalized odds if sensitive column provided
    if sensitive_column is not None and sensitive_column in X_test.columns:
        sens = X_test[sensitive_column].reset_index(drop=True)
        sub_perf = subgroup_performance(sens, y_test.reset_index(drop=True), preds, y_prob=proba)
        eqodds = equalized_odds_check(sens, y_test.reset_index(drop=True), preds)
        report.update({"subgroup_performance": sub_perf, "equalized_odds": eqodds})
    # Create reports dir
    Path("reports").mkdir(exist_ok=True, parents=True)

    # Calibration plot
    calib_path = os.path.join("reports", "calibration.png")
    plot_calibration_curve(y_test, proba, calib_path)

    # Start MLflow run for evaluation
    mlflow.set_experiment(meta.get("model_name", "evaluation"))
    with mlflow.start_run(run_name=f"eval_{meta.get('model_name','model')}"):
        # log metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))
        # log sensitive column info if present
        if sensitive_column is not None and 'sub_perf' in locals():
            mlflow.log_param("sensitive_column", sensitive_column)
            mlflow.log_metric("n_subgroups", len(sub_perf))

        # Log artifacts
        mlflow.log_artifact(calib_path)

        # Optional comparison with another model
        comparisons = {}
        if compare_model_path:
            bundle2 = joblib.load(compare_model_path)
            model2 = bundle2["pipeline"]
            proba2 = model2.predict_proba(X_test)[:, 1]
            preds2 = (proba2 >= 0.5).astype(int)

            # DeLong test for AUCs
            auc1, auc2, z, pvalue = delong_roc_test(y_test, proba, proba2)
            comparisons["delong"] = {"auc1": auc1, "auc2": auc2, "z": z, "pvalue": pvalue}

            # McNemar test on paired predictions
            p_mcnemar, stat = mcnemar_test(y_test, preds, preds2)
            comparisons["mcnemar"] = {"pvalue": p_mcnemar, "stat": stat}

            # Log comparison metrics
            mlflow.log_metric("auc_compare_pvalue", float(pvalue) if pvalue is not None else float('nan'))
            mlflow.log_metric("mcnemar_pvalue", float(p_mcnemar) if p_mcnemar is not None else float('nan'))

        report.update({"comparisons": comparisons})

        # Save report and exit
        Path("reports/last_eval.json").write_text(json.dumps(report, indent=2))
        mlflow.log_artifact("reports/last_eval.json")

    print(json.dumps(report, indent=2))
    return report
 
if __name__ == "__main__":
    # Backwards compatible CLI, optional second model for comparison
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join("app", "model.joblib"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare", default=None, help="Path to another model.joblib to compare against")
    ap.add_argument("--sensitive", default=None, help="Name of sensitive column in features to run subgroup fairness checks")
    args = ap.parse_args()
    evaluate(args.model, seed=args.seed, compare_model_path=args.compare, sensitive_column=args.sensitive)