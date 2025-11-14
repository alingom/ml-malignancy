import argparse, os, sys, json
from pathlib import Path
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

sys.path.append(os.path.dirname(__file__))
from data import load_dataset
from models import get_model

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import joblib
 
def plot_roc_curve(y_true, y_pred, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()
    return output_path

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()
    return output_path

def train_and_save(model_name: str,
                   out_path: str,
                   seed: int = 42,
                   test_size: float = 0.2,
                   cv_n_splits: int = 5,
                   experiment_name: str = "malignancy_prediction"):
    """Train a model and log experiment info to MLflow.

    Parameters come from Hydra config when used via the Hydra entrypoint.
    """
    mlflow.set_experiment(experiment_name)

    # Create a unique run name with timestamp
    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Log basic parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("seed", seed)
        mlflow.log_param("timestamp", datetime.now().isoformat())
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("cv_n_splits", cv_n_splits)

        # Load and log dataset info
        X, y, target_names = load_dataset()
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_classes", len(target_names))
        mlflow.log_param("class_distribution", str(dict(zip(target_names, np.bincount(y)))))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
    
        # Prepare model
        model = get_model(model_name, random_state=seed)

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"[CV] ROC-AUC: mean={scores.mean():.4f} ± {scores.std():.4f}")
        
        # Log cross-validation metrics
        mlflow.log_metric("cv_roc_auc_mean", scores.mean())
        mlflow.log_metric("cv_roc_auc_std", scores.std())
        for i, score in enumerate(scores):
            mlflow.log_metric(f"cv_fold_{i+1}_roc_auc", score)

        # Train final model
        model.fit(X_train, y_train)

        # Get predictions
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        test_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"[TEST] ROC-AUC: {test_auc:.4f}")

        # Log test metrics
        mlflow.log_metric("test_roc_auc", test_auc)

        # Generate and log plots
        # Save plots into the original cwd so they are easy to find when running under Hydra
        try:
            orig_cwd = get_original_cwd()
        except Exception:
            # Hydra not initialized (e.g. when called directly in tests) -> use current cwd
            orig_cwd = os.getcwd()
        roc_plot_path = os.path.join(orig_cwd, "roc_curve.png")
        cm_plot_path = os.path.join(orig_cwd, "confusion_matrix.png")
        plot_roc_curve(y_test, y_pred_proba[:, 1], roc_plot_path)
        plot_confusion_matrix(y_test, y_pred, cm_plot_path)
        mlflow.log_artifact(roc_plot_path)
        mlflow.log_artifact(cm_plot_path)

        # Log classification report
        report = classification_report(y_test, y_pred, target_names=target_names)
        # write report into original cwd for consistency when running under Hydra
        try:
            report_cwd = get_original_cwd()
        except Exception:
            report_cwd = os.getcwd()
        report_path = os.path.join(report_cwd, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Log model parameters
        model_params = model.get_params()
        mlflow.log_params({f"model_{k}": str(v) for k, v in model_params.items()})

        # Log the model in MLflow
        mlflow.sklearn.log_model(model, "model")
 
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
 
    # Resolve output path to original working directory when running under Hydra
    try:
        orig_cwd = get_original_cwd()
    except Exception:
        orig_cwd = os.getcwd()

    # If out_path is relative, place it under orig_cwd; otherwise keep absolute
    out_path_resolved = out_path if os.path.isabs(out_path) else os.path.join(orig_cwd, out_path)
    out = Path(out_path_resolved)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "meta": meta}, out)
    (out.parent / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] Saved model to {out}")
    return meta
 
@hydra.main(config_path="../conf", config_name="config")
def hydra_main(cfg: DictConfig):
    """Hydra entrypoint. Use `python -m src.train` to run with Hydra config.

    Example overrides:
      python -m src.train model.name=rf training.seed=123 data.test_size=0.25
    """
    # Map config values
    model_name = cfg.model.name
    seed = int(cfg.training.seed)
    test_size = float(cfg.data.test_size)
    cv_n_splits = int(cfg.training.cv_n_splits)
    out_path = cfg.output.path
    experiment_name = cfg.experiment.name

    # Call the existing training function
    meta = train_and_save(
        model_name=model_name,
        out_path=out_path,
        seed=seed,
        test_size=test_size,
        cv_n_splits=cv_n_splits,
        experiment_name=experiment_name,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    # Run via Hydra entrypoint (config-driven). Example overrides:
    # python -m src.train model.name=rf training.seed=123 data.test_size=0.25
    hydra_main()