# Mini-Project-1
# What drives a Telco customer's monthly bill, and do big spenders churn more?

**Live app:** http://localhost:8501/

**Dataset:** Telco Customer Churn, https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download

**Target:** MonthlyCharges (predicted with a constant model and OLS with one-hot encoded InternetService)

## Faithfulness & Granularity
- Each row is one customer...
- Scope: ...
- Strange values: ...
- Duplicates: none

## Key finding
Holding tenure constant, fiber optic customers are predicted to pay about $33.60 more per month than DSL customers.

## Run locally
streamlit run app.py