import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import base64

# Page Configuration
st.set_page_config(
    page_title="Breast Cancer - Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4169E1;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1E90FF;
        font-size: 2.2rem !important;
        font-weight: 600;
    }
    h2 {
        color: #2c3e50;
        font-size: 1.8rem !important;
        font-weight: 500;
    }
    h3 {
        color: #34495e;
        font-size: 1.4rem !important;
        font-weight: 500;
    }
    .small-font {
        font-size: 0.9rem !important;
    }
    .stAlert {
        padding: 0.5rem !important;
        font-size: 0.9rem !important;
    }
    .css-1v0mbdj.etr89bj1 {
        margin-top: -60px;
    }
    .css-10trblm.e16nr0p30 {
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Helpers
@st.cache_resource
def load_model(model_path: str = "app/model.joblib"):
    try:
        bundle = joblib.load(model_path)
        return bundle
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_meta(meta_path: str = "app/model_meta.json"):
    p = Path(meta_path)
    if p.exists():
        return json.loads(p.read_text())
    return None

@st.cache_resource
def load_report(report_path: str = "reports/last_eval.json"):
    p = Path(report_path)
    if p.exists():
        return json.loads(p.read_text())
    return None

@st.cache_resource
def load_example(example_path: str = "examples/one.json"):
    p = Path(example_path)
    if p.exists():
        data = json.loads(p.read_text())
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data
    return None


def predict_dataframe(pipeline, df: pd.DataFrame):
    proba = pipeline.predict_proba(df)[:, 1]
    preds = pipeline.predict(df)
    out = df.copy()
    out["prediction"] = preds
    out["probability"] = proba
    return out


def display_metric_card(title, value, unit=""):
    st.markdown(f"""
    <div class="metric-card">
        <p style='margin-bottom: 0.2rem; font-weight: 600; color: #2c3e50;'>{title}</p>
        <h2 style='margin: 0; color: #1E90FF;'>{value}{unit}</h2>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.title("🏥 Breast Cancer Diagnosis System")
    st.markdown("### Medical Decision Support System", unsafe_allow_html=True)

    meta = load_meta()
    report = load_report()
    example = load_example()
    bundle = load_model()

    if bundle is None:
        st.warning("⚠️ Model not found. First run training with `python -m src.train`")
        return

    model = bundle.get("pipeline")

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        if meta:
            cols = st.columns(2)
            with cols[0]:
                display_metric_card("Model", meta.get('model_name', 'unknown'))
            with cols[1]:
                display_metric_card("Samples", meta.get('n_train', 'n/a'))
            
            if meta.get('cv_roc_auc_mean'):
                cols = st.columns(2)
                with cols[0]:
                    display_metric_card("ROC-AUC CV", f"{meta.get('cv_roc_auc_mean'):.3f}")
                with cols[1]:
                    display_metric_card("Test ROC-AUC", f"{meta.get('test_roc_auc'):.3f}")

        st.markdown("---")
        st.header("📤 Input Data")
        uploaded = st.file_uploader("Upload a CSV for batch predictions", type=["csv"])
        st.markdown("<p class='small-font'>or use the form on the right for single prediction.</p>", unsafe_allow_html=True)

    # Main layout: left metrics, right prediction
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📈 Evaluation Report")
        if report:
            # Show metrics
            metrics = report.get("metrics", {})
            st.subheader("Main Metrics")
            metrics_to_show = {
                "ROC-AUC": metrics.get("roc_auc", "N/A"),
                "Precision": metrics.get("precision", "N/A"),
                "Recall": metrics.get("recall", "N/A"),
                "F1-score": metrics.get("f1", "N/A")
            }
            cols = st.columns(len(metrics_to_show))
            for col, (name, value) in zip(cols, metrics_to_show.items()):
                with col:
                    display_metric_card(name, f"{value:.3f}")

            # Show top importances
            st.subheader("🎯 Important Features")
            imps = report.get("top_permutation_importances", [])
            if imps:
                df_imp = pd.DataFrame(imps, columns=["feature", "importance"])
                st.dataframe(df_imp.style.background_gradient(subset=['importance'], cmap='Reds'), 
                           use_container_width=True, height=200)

            # Fairness / subgroup info
            if "subgroup_performance" in report:
                st.subheader("Subgroup Performance")
                st.json(report.get("subgroup_performance"))
            if "equalized_odds" in report:
                st.subheader("Equalized Odds Check")
                st.json(report.get("equalized_odds"))

            # Show plots if available
            st.subheader("📊 Visualizations")
            cols = st.columns(2)
            with cols[0]:
                calib = Path("reports/calibration.png")
                if calib.exists():
                    st.image(str(calib), caption="Calibration Curve")
            with cols[1]:
                roc = Path("reports/roc.png")
                if roc.exists():
                    st.image(str(roc), caption="ROC Curve")
        else:
            st.info("ℹ️ No evaluation report found. Run `python src/evaluate.py`")

        st.markdown("---")
        st.header("Batch Predictions")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.write("Uploaded data preview:")
                st.dataframe(df.head())

                # Try to predict
                if st.button("Run batch predictions"):
                    try:
                        out = predict_dataframe(model, df)
                        st.success("Predictions computed")
                        st.dataframe(out.head())
                        csv = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Download predictions (CSV)", csv, file_name="predictions.csv")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")
        else:
            st.info("No CSV uploaded. Use the manual input on the right or upload a CSV with the model features.")

    with col2:
        st.header("🎯 Single Prediction")
        features = meta.get("features") if meta else None
        defaults = example if example else {}

        if not features:
            st.warning("⚠️ Feature list not available")
        else:
            prediction_result = None
            
            with st.form(key="manual_input"):
                st.subheader("Patient Characteristics")
                values = {}
                # Group related features
                feature_groups = {
                    "Average Measurements": [f for f in features if f.startswith("mean")],
                    "Standard Error": [f for f in features if f.endswith("error")],
                    "Worst Measurements": [f for f in features if f.startswith("worst")]
                }

                for group_name, group_features in feature_groups.items():
                    st.markdown(f"<h4 style='font-size: 1rem; color: #2c3e50;'>{group_name}</h4>", unsafe_allow_html=True)
                    for feat in group_features:
                        default_val = defaults.get(feat) if defaults else None
                        try:
                            v = st.number_input(
                                feat,
                                value=float(default_val) if default_val is not None else 0.0,
                                format="%f",
                                help=f"Enter the value for {feat}"
                            )
                        except:
                            v = 0.0
                        values[feat] = v

                predict_btn = st.form_submit_button(label="🔍 Make Prediction")
                if predict_btn:
                    df = pd.DataFrame([values])
                    try:
                        out = predict_dataframe(model, df)
                        pred = int(out['prediction'].iloc[0])
                        prob = float(out['probability'].iloc[0])
                        prediction_result = {
                            'prediction': pred,
                            'probability': prob,
                            'data': out
                        }
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
            
            # Display result and download button outside the form
            if prediction_result is not None:
                pred = prediction_result['prediction']
                prob = prediction_result['probability']
                out = prediction_result['data']
                
                # Display result with custom style
                result_color = "#ADD8E6" if pred == 0 else "#87CEEB"
                result_text = "Benign" if pred == 0 else "Malignant"
                st.markdown(f"""
                <div style='background-color: {result_color}; padding: 1.5rem; border-radius: 10px; color: #003366;'>
                    <h3 style='margin: 0; color: #003366;'>Result:</h3>
                    <p style='font-size: 1.2rem; margin: 0.5rem 0;'>
                        Diagnosis: <strong>{result_text}</strong>
                    </p>
                    <p style='font-size: 1rem; margin: 0;'>
                        Probability: <strong>{prob:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Prediction (CSV)",
                    csv,
                    file_name="prediction_single.csv",
                    mime="text/csv"
                )

        st.markdown("---")
        st.write("Tips:")
        st.markdown("- Use a CSV with the exact feature names listed in `app/model_meta.json` for batch predictions.")
        st.markdown("- You can generate an example input with `examples/one.json`.")

    st.sidebar.markdown("---")
    st.sidebar.write("Streamlit demo to inspect metrics and run predictions.")

if __name__ == '__main__':
    main()
