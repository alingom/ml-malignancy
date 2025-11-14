import os
import argparse
import time
import joblib
import mlflow
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import sys

# Ensure src is on sys.path when run as module
sys.path.append(os.path.dirname(__file__))
from data import load_dataset
from evaluate import evaluate
from monitoring import detect_drift
from pathlib import Path


def run_check(model_path, incoming_path=None, baseline_path=None, sensitive_column: str = None):
    """Run evaluation and drift detection once and log to MLflow."""
    # Load model
    bundle = joblib.load(model_path)
    model = bundle["pipeline"]
    meta = bundle.get("meta", {})

    # Load baseline data
    if baseline_path and os.path.exists(baseline_path):
        import pandas as pd
        baseline = pd.read_csv(baseline_path)
    else:
        X, y, _ = load_dataset()
        baseline = X

    # Load incoming data (if not provided, reuse test split from load_dataset)
    if incoming_path and os.path.exists(incoming_path):
        import pandas as pd
        incoming = pd.read_csv(incoming_path)
    else:
        # fallback to using a fresh split from built-in dataset
        X, y, _ = load_dataset()
        incoming = X

    # Run evaluation (will log to MLflow inside evaluate)
    eval_report = evaluate(model_path, sensitive_column=sensitive_column)

    # Run drift detection
    drift_report = detect_drift(baseline, incoming, features=baseline.columns.tolist())

    # Log drift report to MLflow
    mlflow.set_experiment(meta.get("model_name", "monitoring"))
    with mlflow.start_run(run_name=f"monitor_{meta.get('model_name','model')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params({"model_path": model_path})
        if sensitive_column:
            mlflow.log_param("sensitive_column", sensitive_column)
        # Log drift counts
        mlflow.log_metric("n_drift_features", len(drift_report.get("drift_features", [])))
        # Save drift report
        import json
        reports_dir = Path(os.getcwd()) / "reports"
        reports_dir.mkdir(exist_ok=True)
        report_path = reports_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        # Convert numpy types to native python types for JSON serialization
        import json
        import numpy as _np

        def _to_native(o):
            if isinstance(o, dict):
                return {k: _to_native(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_to_native(v) for v in o]
            if isinstance(o, _np.generic):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
            return o

        serializable = _to_native(drift_report)
        with open(report_path, "w") as f:
            json.dump(serializable, f, indent=2)
        mlflow.log_artifact(str(report_path))

    return eval_report, drift_report


def main(args):
    if args.once:
        run_check(args.model, incoming_path=args.incoming, baseline_path=args.baseline, sensitive_column=args.sensitive)
        return
        return

    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: run_check(args.model, incoming_path=args.incoming, baseline_path=args.baseline, sensitive_column=args.sensitive), 'interval', minutes=args.interval)
    scheduler.start()
    print(f"Started monitoring scheduler: running every {args.interval} minutes. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.shutdown()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join('app', 'model.joblib'))
    ap.add_argument('--incoming', default=None, help='CSV file with incoming data to check for drift')
    ap.add_argument('--baseline', default=None, help='CSV file with baseline data (training)')
    ap.add_argument('--interval', type=int, default=1440, help='Interval in minutes between checks')
    ap.add_argument('--once', action='store_true', help='Run once and exit')
    ap.add_argument('--sensitive', default=None, help='Name of sensitive column to run subgroup fairness checks during evaluation')
    args = ap.parse_args()
    main(args)
