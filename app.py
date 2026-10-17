import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="AI Model Risk & Audit Toolkit",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ AI Model Risk & Audit Assessment Toolkit")
st.caption(
    "Prototype for model validation, input-drift screening, governance assessment, "
    "and simulated audit-control testing. Uses synthetic financial data."
)

FEATURES = [
    "transaction_count",
    "avg_transaction_value",
    "days_sales_outstanding",
    "prior_month_revenue",
]

with st.sidebar:
    st.header("Assessment settings")
    seed = st.number_input("Random seed", min_value=1, max_value=99999, value=42, step=1)
    drift_strength = st.slider("Current-period input drift", 0.0, 2.0, 0.8, 0.1)
    error_strength = st.slider("Current-period outcome noise", 0.5, 3.0, 1.4, 0.1)
    drift_alpha = st.selectbox("Drift test significance level", [0.01, 0.05, 0.10], index=1)
    st.info("Adjust the settings to simulate different risk scenarios.")

rng = np.random.default_rng(int(seed))

def make_period(n, random_generator, drift=0.0, noise_scale=1.0):
    transaction_count = random_generator.normal(900 + 100 * drift, 140 + 20 * drift, n).clip(250, None)
    avg_transaction_value = random_generator.normal(420 + 35 * drift, 65, n).clip(100, None)
    days_sales_outstanding = random_generator.normal(38 + 8 * drift, 7, n).clip(10, 90)
    prior_month_revenue = (
        transaction_count * avg_transaction_value * random_generator.normal(0.96, 0.06, n)
    )
    revenue = (
        1.15 * transaction_count * avg_transaction_value
        - 900 * days_sales_outstanding
        + 0.22 * prior_month_revenue
        + random_generator.normal(0, 18000 * noise_scale, n)
    )
    return pd.DataFrame({
        "transaction_count": transaction_count,
        "avg_transaction_value": avg_transaction_value,
        "days_sales_outstanding": days_sales_outstanding,
        "prior_month_revenue": prior_month_revenue,
        "actual_revenue": revenue,
    })

train_df = make_period(900, rng, drift=0.0, noise_scale=1.0)
prior_df = make_period(250, rng, drift=0.1, noise_scale=1.0)
current_df = make_period(250, rng, drift=float(drift_strength), noise_scale=float(error_strength))

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), FEATURES)
    ],
    remainder="drop",
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])
model.fit(train_df[FEATURES], train_df["actual_revenue"])

def evaluate(df):
    predictions = model.predict(df[FEATURES])
    actual = df["actual_revenue"].to_numpy()
    return {
        "MAE": float(mean_absolute_error(actual, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predictions))),
        "R²": float(r2_score(actual, predictions)),
        "predictions": predictions,
    }

prior_eval = evaluate(prior_df)
current_eval = evaluate(current_df)
mae_change = (current_eval["MAE"] - prior_eval["MAE"]) / max(prior_eval["MAE"], 1) * 100

def drift_results(reference, current, alpha):
    rows = []
    for col in FEATURES:
        statistic, p_value = ks_2samp(reference[col], current[col])
        rows.append({
            "Feature": col.replace("_", " ").title(),
            "KS statistic": float(statistic),
            "p-value": float(p_value),
            "Drift flagged": bool(p_value < alpha),
        })
    return pd.DataFrame(rows)

drift_df = drift_results(train_df, current_df, float(drift_alpha))
flagged_drift = int(drift_df["Drift flagged"].sum())

# Governance checklist: users can update the evidence status interactively.
governance_defaults = pd.DataFrame([
    ("Governance ownership", "Named owner and accountability for the model are documented.", "Govern", "Not evidenced"),
    ("Intended use", "Model purpose, users, limitations, and prohibited uses are documented.", "Map", "Not evidenced"),
    ("Data quality", "Input data quality checks and lineage evidence are maintained.", "Map", "Not evidenced"),
    ("Model validation", "Performance, assumptions, and limitations are independently reviewed.", "Measure", "Not evidenced"),
    ("Ongoing monitoring", "Drift, errors, and relevant KPIs are monitored with escalation thresholds.", "Measure", "Not evidenced"),
    ("Human oversight", "A qualified reviewer can challenge or override material outputs.", "Manage", "Not evidenced"),
    ("Change management", "Model changes are approved, versioned, and logged.", "Manage", "Not evidenced"),
    ("Incident handling", "A documented process exists to investigate and remediate incidents.", "Manage", "Not evidenced"),
], columns=["Assessment area", "Evidence / expectation", "NIST AI RMF function", "Status"])

control_defaults = pd.DataFrame([
    ("Human approval documented for material AI-assisted entries", "Fail", "Approval log missing for sample entry"),
    ("Exception threshold and escalation process defined", "Pass", "Threshold and escalation owner documented"),
    ("Model version and change history retained", "Needs investigation", "Version history incomplete"),
    ("Input data reconciled to approved source", "Pass", "Sample reconciliation agrees"),
    ("Monitoring alerts reviewed and retained", "Fail", "No reviewer sign-off in sample log"),
], columns=["Control", "Test result", "Evidence reference"])

# Use tabs to keep the application compact and easy to navigate.
overview_tab, drift_tab, governance_tab, audit_tab, reports_tab = st.tabs([
    "📊 Overview",
    "🔎 Drift Detection",
    "🛡️ Governance",
    "📋 Audit Controls",
    "📁 Reports",
])

with overview_tab:
    st.subheader("Model performance validation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current-period MAE", f"${current_eval['MAE']:,.0f}")
    m2.metric("Current-period RMSE", f"${current_eval['RMSE']:,.0f}")
    m3.metric("Current-period R²", f"{current_eval['R²']:.3f}")
    m4.metric(
        "MAE change vs prior",
        f"{mae_change:+.1f}%",
        delta=f"{mae_change:+.1f}% vs prior",
        delta_color="inverse",
    )

    compare = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Prior period": [prior_eval["MAE"], prior_eval["RMSE"], prior_eval["R²"]],
        "Current period": [current_eval["MAE"], current_eval["RMSE"], current_eval["R²"]],
    })
    left, right = st.columns(2)
    with left:
        st.markdown("**Prior vs current period**")
        st.dataframe(
            compare.style.format({
                "Prior period": "{:,.3f}",
                "Current period": "{:,.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("**Actual vs predicted revenue**")
        scatter_df = pd.DataFrame({
            "Actual revenue": current_df["actual_revenue"].to_numpy(),
            "Predicted revenue": current_eval["predictions"],
        })
        st.scatter_chart(scatter_df, x="Actual revenue", y="Predicted revenue")
    st.info(
        "Interpretation: lower MAE/RMSE generally indicates smaller prediction errors. "
        "A rise in MAE is a signal to investigate, not automatic proof that the model is unusable."
    )

with drift_tab:
    st.subheader("Input-data drift screening")
    st.write(
        f"Two-sample Kolmogorov–Smirnov tests compare training inputs with current-period inputs. "
        f"Significance level: {drift_alpha:.2f}. A flagged result is a screening signal, not proof of harmful drift."
    )
    st.dataframe(
        drift_df.style.format({"KS statistic": "{:.3f}", "p-value": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    if flagged_drift:
        st.warning(
            f"{flagged_drift} of {len(FEATURES)} input features show statistically significant distribution differences."
        )
    else:
        st.success("No input features were flagged at the selected significance level.")
    st.markdown("**Recommended follow-up**")
    st.write(
        "Investigate changes in source systems, customer or transaction mix, and data-processing rules. "
        "Assess business materiality and model impact before deciding on remediation."
    )

with governance_tab:
    st.subheader("AI governance evidence assessment")
    st.write(
        "Update each status based on evidence available. The checklist is inspired by the NIST AI RMF functions "
        "Govern, Map, Measure, and Manage; it is not an official maturity assessment."
    )
    edited_gov = st.data_editor(
        governance_defaults,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.SelectboxColumn(
                "Status",
                options=["Effective", "Needs improvement", "Not evidenced"],
                required=True,
            )
        },
        disabled=["Assessment area", "Evidence / expectation", "NIST AI RMF function"],
        key="governance_editor",
    )
    effective_count = int((edited_gov["Status"] == "Effective").sum())
    maturity_score = effective_count / len(edited_gov) * 100
    needs_improvement = int((edited_gov["Status"] == "Needs improvement").sum())
    not_evidenced = int((edited_gov["Status"] == "Not evidenced").sum())
    g1, g2, g3 = st.columns(3)
    g1.metric("Evidence maturity score", f"{maturity_score:.0f}%")
    g2.metric("Needs improvement", needs_improvement)
    g3.metric("Not evidenced", not_evidenced)
    st.caption(
        "Prototype score = percentage of checklist areas marked Effective. It is not an official NIST or ISO rating. "
        "'Not evidenced' means no supporting evidence was entered in this demo."
    )

with audit_tab:
    st.subheader("Simulated audit controls testing")
    st.warning(
        "The default results below are illustrative placeholders. In real work, conclusions must be based on actual evidence."
    )
    edited_controls = st.data_editor(
        control_defaults,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Test result": st.column_config.SelectboxColumn(
                "Test result",
                options=["Pass", "Fail", "Needs investigation"],
                required=True,
            )
        },
        key="controls_editor",
    )
    pass_count = int((edited_controls["Test result"] == "Pass").sum())
    fail_count = int((edited_controls["Test result"] == "Fail").sum())
    investigate_count = int((edited_controls["Test result"] == "Needs investigation").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Pass", pass_count)
    c2.metric("Fail", fail_count)
    c3.metric("Needs investigation", investigate_count)
    st.markdown("**Testing approach**")
    st.write(
        "For each control, record the expected control, sample selected, evidence inspected, result, exception details, "
        "reviewer, and remediation owner. Do not infer that a control operated effectively without sufficient evidence."
    )

with reports_tab:
    st.subheader("Assessment findings and report exports")
    # Read current widget state, falling back to defaults if a tab has not been interacted with yet.
    edited_gov = st.session_state.get("governance_editor", governance_defaults)
    edited_controls = st.session_state.get("controls_editor", control_defaults)
    if not isinstance(edited_gov, pd.DataFrame):
        edited_gov = governance_defaults
    if not isinstance(edited_controls, pd.DataFrame):
        edited_controls = control_defaults

    not_evidenced = int((edited_gov["Status"] == "Not evidenced").sum())
    fail_count = int((edited_controls["Test result"] == "Fail").sum())
    investigate_count = int((edited_controls["Test result"] == "Needs investigation").sum())

    findings = []
    if flagged_drift:
        findings.append({
            "Priority": "High" if flagged_drift >= 2 else "Medium",
            "Finding": f"{flagged_drift} input feature(s) flagged for distribution drift",
            "Recommended action": "Investigate source/process changes, assess model impact, and document the disposition.",
            "Reference": "Model monitoring / data quality",
        })
    if mae_change > 15:
        findings.append({
            "Priority": "High",
            "Finding": f"Current-period MAE increased by {mae_change:.1f}% versus prior period",
            "Recommended action": "Investigate root cause, compare against an approved tolerance, and consider additional review.",
            "Reference": "Model performance monitoring",
        })
    if fail_count:
        findings.append({
            "Priority": "High",
            "Finding": f"{fail_count} simulated control(s) failed",
            "Recommended action": "Obtain missing evidence, assess compensating controls, assign an owner, and track remediation.",
            "Reference": "Audit controls testing",
        })
    if investigate_count:
        findings.append({
            "Priority": "Medium",
            "Finding": f"{investigate_count} simulated control(s) require investigation",
            "Recommended action": "Request supporting evidence, expand testing if appropriate, and document the conclusion.",
            "Reference": "Audit evidence",
        })
    if not_evidenced:
        findings.append({
            "Priority": "Medium",
            "Finding": f"{not_evidenced} governance area(s) have no evidence recorded",
            "Recommended action": "Request ownership, validation, monitoring, data-lineage, and change-management evidence.",
            "Reference": "AI governance",
        })
    if not findings:
        findings.append({
            "Priority": "Low",
            "Finding": "No issues triggered by current demo settings",
            "Recommended action": "Confirm results against defined tolerances and retain supporting evidence.",
            "Reference": "General",
        })

    findings_df = pd.DataFrame(findings)
    st.dataframe(findings_df, use_container_width=True, hide_index=True)

    assessment = {
        "assessment_timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "prototype_note": "Synthetic data and simulated control results; not a formal audit opinion.",
        "settings": {
            "random_seed": int(seed),
            "drift_strength": float(drift_strength),
            "current_period_noise_strength": float(error_strength),
            "drift_significance_level": float(drift_alpha),
        },
        "model_metrics": {
            "prior_period": {k: v for k, v in prior_eval.items() if k != "predictions"},
            "current_period": {k: v for k, v in current_eval.items() if k != "predictions"},
            "mae_change_percent": float(mae_change),
        },
        "drift_results": drift_df.to_dict(orient="records"),
        "governance_assessment": edited_gov.to_dict(orient="records"),
        "control_tests": edited_controls.to_dict(orient="records"),
        "findings": findings_df.to_dict(orient="records"),
    }
    metrics_df = pd.DataFrame([
        {"Section": "Model metrics", "Item": "Prior-period MAE", "Result": prior_eval["MAE"]},
        {"Section": "Model metrics", "Item": "Current-period MAE", "Result": current_eval["MAE"]},
        {"Section": "Model metrics", "Item": "Prior-period RMSE", "Result": prior_eval["RMSE"]},
        {"Section": "Model metrics", "Item": "Current-period RMSE", "Result": current_eval["RMSE"]},
        {"Section": "Model metrics", "Item": "Prior-period R²", "Result": prior_eval["R²"]},
        {"Section": "Model metrics", "Item": "Current-period R²", "Result": current_eval["R²"]},
    ])
    st.download_button(
        "⬇️ Download metrics summary (CSV)",
        data=metrics_df.to_csv(index=False),
        file_name="model_risk_metrics_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Download full assessment report (JSON)",
        data=json.dumps(assessment, indent=2, default=str),
        file_name="ai_model_risk_assessment.json",
        mime="application/json",
        use_container_width=True,
    )

with st.expander("Methodology and limitations"):
    st.markdown("""
    - **Synthetic scenario:** All financial records and audit-control results are simulated.
    - **Model:** Linear regression predicts revenue using transaction count, average transaction value,
      days sales outstanding, and prior-month revenue.
    - **Performance metrics:** MAE and RMSE measure prediction error; R² measures fit on the evaluation sample.
    - **Drift:** Two-sample Kolmogorov–Smirnov tests screen numeric input distributions. Statistical significance does
      not establish material business impact; consider effect size, business thresholds, and model performance.
    - **Governance:** Checklist language is inspired by NIST AI RMF functions Govern, Map, Measure, and Manage.
      This is not an official NIST assessment or ISO/IEC 42001 conformity evaluation.
    - **Controls:** Default test outcomes are illustrative placeholders and must be replaced with evidence-based results
      in any real engagement. This application does not provide an audit opinion.
    """)
