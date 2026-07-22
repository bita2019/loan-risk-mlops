"""
data_quality.py

Schema validation and data quality checks, run before anything touches
the model. This is deliberately kept simple and dependency-free (no
Great Expectations, no external framework) so the checks are easy to
read and reason about — but the categories of check (schema, nulls,
range/sanity, referential) mirror what a production pipeline would run.
"""

from dataclasses import dataclass, field

import pandas as pd

EXPECTED_SCHEMA = {
    "applicant_id": "object",
    "income": "float64",
    "credit_score": "float64",
    "loan_amount": "float64",
    "employment_years": "float64",
    "debt_to_income": "float64",
    "previous_defaults": "int64",
    "default": "int64",
}

RANGE_CHECKS = {
    "income": (0, 500_000),
    "credit_score": (300, 850),
    "loan_amount": (0, 200_000),
    "employment_years": (0, 60),
    "debt_to_income": (0, 1),
    "previous_defaults": (0, 20),
}


@dataclass
class DataQualityReport:
    n_rows: int
    schema_issues: list = field(default_factory=list)
    null_counts: dict = field(default_factory=dict)
    range_violations: dict = field(default_factory=dict)
    duplicate_ids: int = 0

    @property
    def passed(self) -> bool:
        return (
            not self.schema_issues
            and not self.range_violations
            and self.duplicate_ids == 0
        )

    def summary(self) -> str:
        lines = [f"Data quality report — {self.n_rows} rows"]
        lines.append(f"  Schema issues: {self.schema_issues or 'none'}")
        lines.append(f"  Duplicate applicant_ids: {self.duplicate_ids}")
        lines.append("  Null counts:")
        for col, count in self.null_counts.items():
            if count:
                lines.append(f"    {col}: {count}")
        lines.append(f"  Range violations: {self.range_violations or 'none'}")
        lines.append(f"  Overall: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def run_checks(df: pd.DataFrame) -> DataQualityReport:
    report = DataQualityReport(n_rows=len(df))

    # 1. Schema check — right columns, right types
    for col, expected_dtype in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            report.schema_issues.append(f"missing column: {col}")
        elif str(df[col].dtype) != expected_dtype and not (
            expected_dtype == "int64" and str(df[col].dtype) in ("int32", "int64")
        ):
            # allow float64 columns that are actually all-null-free ints etc.
            if not (expected_dtype == "float64" and pd.api.types.is_numeric_dtype(df[col])):
                report.schema_issues.append(
                    f"{col}: expected {expected_dtype}, got {df[col].dtype}"
                )

    # 2. Null counts — reported, not auto-dropped, so the caller decides policy
    report.null_counts = {col: int(df[col].isna().sum()) for col in df.columns}

    # 3. Range / sanity checks
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        violations = df[(df[col] < lo) | (df[col] > hi)]
        if len(violations):
            report.range_violations[col] = len(violations)

    # 4. Duplicate primary key
    if "applicant_id" in df.columns:
        report.duplicate_ids = int(df["applicant_id"].duplicated().sum())

    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a simple, explicit cleaning policy after the report has been reviewed.

    Policy: drop rows with a null in any required numeric feature, rather
    than silently imputing — for a credit-risk-style dataset, silently
    filling in income is exactly the kind of thing that should be a
    deliberate, reviewed decision, not a default.
    """
    required = ["income", "credit_score", "loan_amount", "employment_years", "debt_to_income"]
    before = len(df)
    df_clean = df.dropna(subset=required).reset_index(drop=True)
    dropped = before - len(df_clean)
    if dropped:
        print(f"Dropped {dropped} rows with nulls in required columns ({dropped/before:.2%})")
    return df_clean


if __name__ == "__main__":
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "data" / "loans_raw.csv"
    df = pd.read_csv(path)
    report = run_checks(df)
    print(report.summary())
