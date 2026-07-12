# Loan Default Risk — Data Quality + scikit-learn + MLflow

Small project I built to get hands-on with MLflow and the ML lifecycle,
since that side of things was mostly conceptual for me before this
(my background is data engineering, not ML).

Uses a synthetic loan dataset (income, credit score, loan amount,
employment years, debt-to-income, previous defaults) to predict default
risk. Data's made up so the project is self-contained.

## What it does

1. Generates the synthetic data
2. Runs data quality checks (schema, nulls, ranges, duplicates) before
   any of it touches a model
3. Trains two scikit-learn models (logistic regression + random forest)
4. Logs everything to MLflow — params, metrics, the model itself
5. Registers the best model in the MLflow Model Registry

mermaid
flowchart LR
A[generate_data.py] --> B[(loans_raw.csv)]
B --> C[data_quality checks]
C --> D[train/test split]
D --> E[LogisticRegression]
D --> F[RandomForest]
E --> G[MLflow tracking]
F --> G
G --> H[Model Registry]
