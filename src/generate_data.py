"""
generate_data.py

Generates a loan-default dataset. This keeps the project
self-contained (no external download needed) while staying close to
a realistic credit-risk domain.

In a real-world version of this project, this step would be replaced
by pulling from a real source (e.g. Kaggle's "Loan Default" dataset,
or a production data warehouse table).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_ROWS = 5000

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "loans_raw.csv"


def generate_loans(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    income = rng.normal(45000, 15000, n_rows).clip(12000, 200000)
    credit_score = rng.normal(650, 80, n_rows).clip(300, 850)
    loan_amount = rng.normal(15000, 8000, n_rows).clip(1000, 60000)
    employment_years = rng.exponential(4, n_rows).clip(0, 40)
    debt_to_income = rng.beta(2, 5, n_rows) * 0.6
    previous_defaults = rng.poisson(0.3, n_rows).clip(0, 5)

    income = income.astype(float)

    # Default probability is a function of the risk features above, plus
    # noise — not a hand-picked label, so the model has something genuine
    # to learn. Computed BEFORE null injection so the label itself is
    # always well-defined.
    risk_score = (
        1.0
        + -0.00003 * income
        + -0.004 * credit_score
        + 0.00002 * loan_amount
        + -0.04 * employment_years
        + 2.5 * debt_to_income
        + 0.5 * previous_defaults
    )
    prob_default = 1 / (1 + np.exp(-(risk_score + rng.normal(0, 0.6, n_rows))))
    default = rng.binomial(1, prob_default)

    # A handful of rows get nulls injected AFTER the label is computed —
    # this gives the data-quality step something real to catch.
    null_idx = rng.choice(n_rows, size=int(n_rows * 0.01), replace=False)
    income[null_idx] = np.nan

    df = pd.DataFrame(
        {
            "applicant_id": [f"APP{i:06d}" for i in range(n_rows)],
            "income": income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
            "debt_to_income": debt_to_income,
            "previous_defaults": previous_defaults,
            "default": default,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_loans()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"Nulls in income: {df['income'].isna().sum()}")
