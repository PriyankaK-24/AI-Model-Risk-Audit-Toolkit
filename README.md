# AI Model Risk & Audit Assessment Toolkit

An interactive Python and Streamlit application for assessing machine learning model performance, detecting input-data drift, evaluating AI governance evidence, and documenting simulated audit-control findings in a financial reporting scenario.

## Overview

Organizations increasingly rely on AI and machine learning for business and financial decisions. However, model performance can deteriorate, input data can change, and governance or monitoring controls may lack sufficient evidence.

This project demonstrates a lightweight assessment workflow that brings together model validation, statistical drift screening, AI governance checks, and audit-control testing in one interactive dashboard.

## Key Features

- **Model Validation:** Evaluate a financial revenue prediction model using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².
- **Performance Comparison:** Compare model results across prior and current reporting periods to identify potential deterioration.
- **Data Drift Detection:** Apply two-sample Kolmogorov–Smirnov tests to screen for distribution changes in model input features.
- **AI Governance Assessment:** Evaluate evidence across governance ownership, intended use, data quality, validation, monitoring, human oversight, change management, and incident handling.
- **Audit Controls Testing:** Record simulated control outcomes for approvals, exception monitoring, data reconciliation, and model-change documentation.
- **Risk Reporting:** Generate findings, remediation recommendations, and downloadable CSV and JSON reports.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- SciPy
- Streamlit

## Methodology

### 1. Model Performance Evaluation
A linear regression model predicts synthetic financial revenue using transaction count, average transaction value, days sales outstanding, and prior-month revenue.

### 2. Statistical Drift Screening
The Kolmogorov–Smirnov two-sample test compares training-period input distributions with current-period inputs. Flagged features are investigated as potential drift risks rather than automatically treated as model failures.

### 3. Governance Assessment
The governance checklist is inspired by the NIST AI Risk Management Framework functions: Govern, Map, Measure, and Manage. Users can update evidence statuses and review a prototype scorecard.

### 4. Audit Controls Testing
Illustrative control tests identify simulated exceptions and support documenting remediation actions. The dashboard is designed to demonstrate the assessment workflow, not replace evidence-based audit procedures.

## Getting Started

### Prerequisites
- Python 3.10 or later
- pip

### Installation

Clone the repository:

```bash
git clone https://github.com/PriyankaK-24/AI-Model-Risk-Audit-Toolkit.git
cd AI-Model-Risk-Audit-Toolkit
```

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Limitations and Responsible Use

- Financial data is synthetic and intended for demonstration.
- Audit-control outcomes are illustrative placeholders, not findings from a real client engagement.
- The governance score is a prototype checklist score, not an official NIST maturity rating or ISO/IEC 42001 certification.
- Statistical drift does not necessarily imply material model degradation.
- This project does not constitute a formal audit opinion or a compliance assessment.

## Future Enhancements

- Add SHAP-based model explainability.
- Introduce configurable performance thresholds and alerting.
- Incorporate evidence uploads and audit-trail tracking.
- Add tests and expand model validation scenarios.

## Author

**Priyanka K.**

Developed as a portfolio project exploring AI model risk, statistical validation, governance assessment, and financial audit controls.
