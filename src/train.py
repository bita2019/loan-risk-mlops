"""
train.py

Loads the (already quality-checked) loan data, trains a scikit-learn
classifier to predict default risk, and logs everything to MLflow:
parameters, metrics, the model artifact itself, and a registered
model version.

Run:
    python src/train.py
Then inspect results with:
    mlflow ui --port 5000
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_quality import clean, run_checks

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "loans_raw.csv"
FEATURE_COLS = [
    "income",
    "credit_score",
    "loan_amount",
    "employment_years",
    "debt_to_income",
    "previous_defaults",
]
TARGET_COL = "default"

# Explicit local SQLite backend — MLflow's plain-filesystem store is in
# maintenance mode as of MLflow 3.x, so a database backend is the
# supported path even for a small local project like this one.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("loan-default-risk")


def load_and_validate() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    report = run_checks(df)
    print(report.summary())

    df = clean(df)
    return df


def train_one_model(model_name: str, model, X_train, X_test, y_train, y_test, params: dict):
    """Train, evaluate, and log a single model as its own MLflow run."""
    with mlflow.start_run(run_name=model_name):
        # --- log the config for this run, so runs are comparable later ---
        mlflow.log_param("model_type", model_name)
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))
        mlflow.log_param("features", ",".join(FEATURE_COLS))

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        # --- metrics: the ones that actually matter for an imbalanced
        # default-prediction task, not just accuracy ---
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.iloc[:5],
            registered_model_name="loan-default-classifier",
        )

        print(f"\n{model_name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return metrics


def main():
    df = load_and_validate()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Train/test split — stratified, since default is the minority class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
    )

    # Two models logged as two separate MLflow runs, so they can be
    # compared side by side in the MLflow UI — this is the "compare
    # experiment runs" behaviour the MLflow course covers.
    # class_weight="balanced" matters here: defaults are ~12% of the data,
    # and an unweighted model happily predicts "no default" for almost
    # everyone and still gets ~88% accuracy — high accuracy, useless model.
    # This is exactly the kind of thing worth catching before shipping.
    train_one_model(
        "logistic_regression",
        LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
        X_train_scaled, X_test_scaled, y_train, y_test,
        params={"C": 1.0, "max_iter": 1000, "scaled_features": True, "class_weight": "balanced"},
    )

    train_one_model(
        "random_forest",
        RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
        ),
        X_train, X_test, y_train, y_test,  # tree models don't need scaling
        params={
            "n_estimators": 200, "max_depth": 6,
            "scaled_features": False, "class_weight": "balanced",
        },
    )

    print("\nDone. Run `mlflow ui` to compare the two runs and inspect the registered model.")


if __name__ == "__main__":
    main()
