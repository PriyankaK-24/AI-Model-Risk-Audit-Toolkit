# AI Model Risk & Audit Assessment Toolkit

A compact Streamlit prototype demonstrating:
- Regression model validation for a synthetic financial forecasting use case
- Prior-period vs current-period performance comparison (MAE, RMSE, R²)
- Input-data drift screening with two-sample Kolmogorov–Smirnov tests
- NIST AI RMF-inspired governance evidence checklist and scorecard
- Simulated audit-control testing and remediation findings
- CSV and JSON report exports

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Launch the app:

   ```bash
   streamlit run app.py
   ```

## Important limitations

This is an educational prototype. Data and audit-control results are synthetic/illustrative, the governance score is not an official maturity rating, and the application is not a formal audit opinion or certification. Replace simulated evidence with approved evidence and documented thresholds for any real use.

## Suggested resume title

AI Model Risk & Audit Assessment Toolkit | Python, Pandas, Scikit-learn, Streamlit, Statistical Analysis
