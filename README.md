# End-to-End ML Case Study: Early Breast Cancer Diagnosis (Classification)
 
Teaching-grade example for **PAU 3102 Research Methods**: problem framing → data →
methodology → experiments & evaluation → deployment → reproducibility.
 
## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train model.name=gb               # trains & saves to app/model.joblib (Hydra-driven)
# Examples of config overrides with Hydra:
# python -m src.train model.name=rf training.seed=123 data.test_size=0.25
python src/evaluate.py                          # prints metrics & saves reports/last_eval.json
python src/predict.py --json examples/one.json  # runs a sample prediction
uvicorn app.main:app --reload                   # run API locally
```
 
 # MLflow UI
 mlflow ui  # Visualisez les résultats avec MLflow at http://localhost:5000

## Data versioning with DVC

This repository includes a minimal DVC pipeline to version model artefacts and make runs reproducible.

Files added for DVC integration:
- `dvc.yaml` - defines a `train` stage that runs the Hydra-driven training command and declares outputs (`app/model.joblib`, `app/model_meta.json`).
- `params.yaml` - parameters that DVC can track across runs (model name, seed, test split, cv folds, output path).
- `.dvcignore` - ignore rules for DVC/git.

How to use DVC locally (one-time setup):

1. Install DVC (you can also install via pip using the project requirements):
```powershell
pip install dvc
```

2. Initialize DVC in your repo (run once):
```powershell
dvc init
```

3. Configure a remote storage for model/artifact pushes (example: S3, GCS, SSH or a local directory):
```powershell
# example local remote
dvc remote add -d myremote C:\path\to\dvc_remote
# or S3: dvc remote add -d myremote s3://my-bucket/path
```

4. Reproduce the `train` stage (runs the command and creates the outputs listed in `dvc.yaml`):
```powershell
dvc repro
```

5. Track and push artifacts to the remote:
```powershell
dvc push
```

Notes:
- Because this project uses the sklearn built-in dataset, there is no large external raw dataset to `dvc add` by default. DVC is useful here to version the produced model artefacts (e.g., `app/model.joblib`) and `model_meta.json` produced by training.
- To pin artifacts to commits, use git to commit the generated `.dvc` metafiles and the `dvc.lock` file (created after `dvc repro`) so each git commit references the exact artifact versions.
- You can add `dvc add <path/to/data>` and commit the generated `<file>.dvc` file to version external datasets if you add external data later.

## Web demo (Streamlit)

A lightweight Streamlit app is included to inspect evaluation metrics and run predictions locally.

Run the app after installing dependencies:

```powershell
# install deps (if not already)
pip install -r requirements.txt

# run streamlit UI
streamlit run app/streamlit_app.py
```

The sidebar allows uploading a CSV for batch predictions. On the right you can enter a single sample using the features saved in `app/model_meta.json`.
