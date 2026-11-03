from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ai_summary import ai_summary_available, build_verified_facts, generate_business_summary, ollama_model, summary_provider
from src.explain import explain_prediction
from src.metrics import load_dataset
from src.predict import load_model, model_schema, predict_customer
from src.recommendations import recommend_actions
from src.segmentation import fit_segments, profile_segments
from src.monitoring import data_quality_summary, numeric_drift

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Explainable Customer Retention Decision-Support System", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .result-card { background: #10170f; border: 1px solid #1f3d20; border-radius: 12px; padding: 24px; }
    .stButton > button { background: #22c55e; color: #08130a; font-weight: 700; width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model(ROOT / "models" / "churn_model.pkl")


model = get_model()
schema = model_schema(model)
feature_order = schema["numeric"] + schema["categorical"]

NUMERIC_RANGES = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 72),
    "MonthlyCharges": (0.0, 200.0),
    "TotalCharges": (0.0, 10000.0),
}

with st.sidebar:
    st.title("Explainable Customer Retention")
    st.caption("Validated prediction, explanation, and retention investigation support.")
    st.divider()
    with st.form("customer_form"):
        user_input = {}
        for column in schema["categorical"]:
            label = column.replace("_", " ")
            user_input[column] = st.selectbox(label, schema["categories"][column])
        for column in schema["numeric"]:
            low, high = NUMERIC_RANGES.get(column, (0.0, 100.0))
            if column == "SeniorCitizen":
                user_input[column] = int(st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No"))
            else:
                label = {"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges (USD)", "TotalCharges": "Total Charges (USD)"}.get(column, column)
                user_input[column] = st.number_input(label, min_value=float(low), max_value=float(high), value=float((low + high) / 2), step=1.0)
        submitted = st.form_submit_button("Predict churn")

if submitted:
    try:
        result = predict_customer(model, user_input)
        input_df = pd.DataFrame([user_input])[feature_order]
        explanation = explain_prediction(model, input_df)
        actions = recommend_actions(user_input, result["probability"], explanation)
        st.session_state["result"] = result
        st.session_state["explanation"] = explanation
        st.session_state["actions"] = actions
        st.session_state["customer"] = user_input
    except ValueError as error:
        st.error(str(error))

st.title("Explainable Customer Retention Decision-Support System")
st.write("The saved churn pipeline remains the source of truth. Explanations and recommendations are separate supporting layers.")

if "result" not in st.session_state:
    st.info("Enter customer information in the sidebar and select **Predict churn** to begin.")
else:
    result = st.session_state["result"]
    explanation = st.session_state["explanation"]
    actions = st.session_state["actions"]
    tabs = st.tabs(["Prediction", "Explanation", "Recommended Actions", "Segments", "Model Performance", "AI Summary", "Monitoring"])

    with tabs[0]:
        left, right = st.columns([0.6, 0.4])
        with left:
            st.subheader("Prediction result")
            st.metric("Estimated churn probability", f"{result['probability']:.1%}")
            st.metric("Risk band", result["risk_band"])
            st.caption("Risk bands are provisional and should be calibrated to retention capacity and intervention cost.")
        with right:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=result["probability"] * 100, number={"suffix": "%"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#22c55e"}}))
            gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(gauge, use_container_width=True)
        st.subheader("Customer input summary")
        st.dataframe(pd.DataFrame({"Feature": list(st.session_state["customer"].keys()), "Value": list(st.session_state["customer"].values())}), hide_index=True, use_container_width=True)

    with tabs[1]:
        st.subheader("Individual model factors")
        chart_df = explanation.sort_values("impact")
        chart = go.Figure(go.Bar(x=chart_df["impact"], y=chart_df["display_feature"], orientation="h", marker_color=["#ef4444" if value > 0 else "#3b82f6" for value in chart_df["impact"]]))
        chart.update_layout(height=420, xaxis_title="SHAP contribution to churn prediction", yaxis_title="Feature")
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(explanation[["display_feature", "impact"]].rename(columns={"display_feature": "Feature", "impact": "Contribution"}), hide_index=True, use_container_width=True)
        st.warning("SHAP contributions describe model behavior for this prediction; they are associations, not proof of causation.")

    with tabs[2]:
        st.subheader("Recommended investigation actions")
        for item in actions:
            st.markdown(f"**{item['priority']} priority — {item['action']}**")
            st.caption(item["reason"])
        st.info("These are transparent investigation suggestions, not guaranteed retention solutions or automatic customer treatments.")

    with tabs[3]:
        st.subheader("Customer segments")
        segment_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        segment_df, segment_labels, segment_scores = None, None, None
        try:
            raw_X, raw_y = load_dataset(ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
            segment_input = raw_X.copy()
            segment_input["Churn"] = raw_y
            _, segmented, scores = fit_segments(segment_input, segment_features)
            st.write(f"Selected k: **{max(scores, key=scores.get)}** based on the highest silhouette score.")
            st.dataframe(profile_segments(segmented).round(4), hide_index=True, use_container_width=True)
            st.caption("Segments are analytical groupings based on selected numeric features; they are not validated natural customer types.")
        except Exception as error:
            st.warning(f"Segmentation is unavailable: {error}")

    with tabs[4]:
        st.subheader("Cross-validation results")
        metrics_path = ROOT / "model_comparison.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            st.dataframe(metrics.round(4), hide_index=True, use_container_width=True)
            st.caption("Metrics were generated with stratified 5-fold cross-validation. The saved model and evaluation environment should be version-pinned before deployment.")
        else:
            st.info("Run the evaluation workflow to populate verified model metrics.")

    with tabs[5]:
        st.subheader("Optional AI-generated business summary")
        st.caption("The language model receives verified risk, explanation, and recommendation facts only; it does not replace the churn model.")
        provider = summary_provider()
        if provider == "ollama":
            st.success(f"Using local Ollama model: {ollama_model()}")
        elif provider == "openai":
            st.info("Using OpenAI because local Ollama is unavailable.")
        else:
            st.info("No external language model is configured. The app will generate a built-in summary from the verified model facts. Add OPENAI_API_KEY in Streamlit Secrets for an AI-written summary.")
        if ai_summary_available() and st.button("Generate business summary"):
            try:
                facts = build_verified_facts(result, explanation, actions)
                with st.spinner("Generating summary from verified facts..."):
                    summary = generate_business_summary(facts)
                st.markdown(f"**Summary**\n\n{summary['summary']}")
                st.markdown("**Risk factors**")
                for factor in summary["risk_factors"]:
                    st.write(f"- {factor}")
                st.markdown("**Recommended next steps**")
                for step in summary["recommended_next_steps"]:
                    st.write(f"- {step}")
                st.warning(summary["caveat"])
                st.caption("AI-generated text summarizes verified model outputs. It does not replace model validation, data validation, or human review.")
            except Exception as error:
                st.error(f"Summary generation failed: {error}")

    with tabs[6]:
        st.subheader("Monitoring prototype")
        reference_X, _ = load_dataset(ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        st.dataframe(data_quality_summary(reference_X), hide_index=True, use_container_width=True)
        st.caption("This prototype reports reference-data quality. Compare it with a later dataset to monitor missingness, category mix, numeric drift, risk drift, and labeled performance.")

st.divider()
st.caption("This tool supports human review and does not establish customer intent, cause, or guaranteed outcomes.")
