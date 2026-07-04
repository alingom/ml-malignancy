# End-to-End ML Case Study: Early Breast Cancer Diagnosis

Teaching-grade machine learning project for binary breast cancer classification. The repo covers data loading, model training, evaluation, model artifact handling, a FastAPI prediction API, and a Streamlit demo.

## Project Structure

```text
app/          FastAPI service, Streamlit UI, and generated model artifacts
conf/         Hydra configuration
examples/     Example prediction payloads
src/          Data, training, evaluation, monitoring, and prediction code
tests/        Pytest suite
```

Generated run outputs such as `mlruns/`, `outputs/`, reports, plots, and local model binaries are ignored by Git. Recreate them locally with the training and evaluation commands below.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Train And Evaluate

Train a model with the default Hydra config:

```bash
python -m src.train
```

Override model and training settings as needed:

```bash
python -m src.train model.name=rf training.seed=123 data.test_size=0.25
```

Evaluate the saved model:

```bash
python -m src.evaluate --model app/model.joblib
```

Run a sample CLI prediction:

```bash
python -m src.predict --json examples/one.json --model app/model.joblib
```

## FastAPI Service

Start the API:

```bash
uvicorn app.main:app --reload
```

Useful endpoints:

```text
GET  /health
GET  /metadata
POST /predict
POST /predict/batch
```

Single prediction payload:

```json
{
  "features": {
    "mean radius": 12.0,
    "mean texture": 15.0
  }
}
```

The real model expects every feature listed in `app/model_meta.json`, which is created by `python -m src.train`.

## Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

The demo reads the saved model and metadata from `app/`, displays evaluation metrics when available, and supports single or batch predictions.

## MLflow And DVC

Start the MLflow UI:

```bash
mlflow ui
```

Reproduce the DVC pipeline:

```bash
dvc repro
```

If you configure a DVC remote, push generated artifacts with:

```bash
dvc push
```

## Development Workflow

```bash
git checkout fix/app-and-tests
pytest
git add .
git commit -m "Fix: add API predictions and cleanup repo"
git push origin fix/app-and-tests
```
