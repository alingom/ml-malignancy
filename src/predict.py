import argparse, os, sys, json
from pathlib import Path
import pandas as pd
import joblib
 
def load_model(path):
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle.get("meta", {})
 
def read_input(json_path=None, csv_path=None):
    if json_path:
        data = json.loads(Path(json_path).read_text())
        return pd.DataFrame(data)
    if csv_path:
        return pd.read_csv(csv_path)
    raise ValueError("Provide --json or --csv input")
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="Path to JSON file (list of records)")
    ap.add_argument("--csv", default=None, help="Path to CSV file")
    ap.add_argument("--model", default=os.path.join("app", "model.joblib"))
    args = ap.parse_args()
 
    X = read_input(args.json, args.csv)
    model, meta = load_model(args.model)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
 
    out = []
    for i in range(len(X)):
        out.append({
            "prediction": int(pred[i]),
            "probability": float(proba[i])
        })
    print(json.dumps(out, indent=2))
 
 