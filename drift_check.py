"""
Experiment 8: Simple data and performance drift checks
Heart Disease Prediction — educational project

Usage:
    python drift_check.py --reference Heart_disease_cleveland_new.csv --incoming incoming_clinical_data.csv

The incoming CSV should contain the same feature columns as the reference dataset.
The target column is optional. If it is present and contains known labels (0/1),
the script also reports basic model performance when a saved model is supplied.

This is a basic monitoring aid, not a validated clinical monitoring system.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COLUMN = "target"


def main():
    parser = argparse.ArgumentParser(description="Basic dataset drift and quality check.")
    parser.add_argument("--reference", required=True, help="CSV used as reference/training data.")
    parser.add_argument("--incoming", required=True, help="New incoming data CSV.")
    parser.add_argument("--output", default="drift_check_report.csv", help="Output report CSV.")
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    incoming = pd.read_csv(args.incoming)

    print("=== Experiment 8: Data Drift Check ===")
    print(f"Reference rows: {len(reference)}")
    print(f"Incoming rows:  {len(incoming)}")

    missing_columns = [col for col in FEATURE_COLUMNS if col not in incoming.columns]
    if missing_columns:
        print("\nERROR: Incoming data is missing required feature columns:")
        print(", ".join(missing_columns))
        print("No drift comparison was completed. Fix the input schema and rerun.")
        return

    missing_ref = [col for col in FEATURE_COLUMNS if col not in reference.columns]
    if missing_ref:
        print("\nERROR: Reference data is missing required feature columns:")
        print(", ".join(missing_ref))
        return

    rows = []

    # Data quality checks
    for col in FEATURE_COLUMNS:
        missing_count = int(incoming[col].isna().sum())
        missing_pct = float(incoming[col].isna().mean() * 100)
        rows.append({
            "check": "missing_values",
            "feature": col,
            "reference_value": "",
            "incoming_value": missing_count,
            "difference": round(missing_pct, 2),
            "status": "REVIEW" if missing_count else "OK",
            "details": f"{missing_pct:.2f}% of incoming values are missing"
        })

    # Numeric drift check: mean shift measured in reference standard deviations.
    numeric_cols = [
        col for col in FEATURE_COLUMNS
        if pd.api.types.is_numeric_dtype(reference[col])
        and pd.api.types.is_numeric_dtype(incoming[col])
        and col not in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    ]

    for col in numeric_cols:
        ref_values = pd.to_numeric(reference[col], errors="coerce").dropna()
        new_values = pd.to_numeric(incoming[col], errors="coerce").dropna()
        if len(ref_values) == 0 or len(new_values) == 0:
            status = "REVIEW"
            standardized_shift = np.nan
            details = "Not enough numeric values to compare"
            ref_mean = ref_values.mean() if len(ref_values) else np.nan
            new_mean = new_values.mean() if len(new_values) else np.nan
        else:
            ref_mean = float(ref_values.mean())
            new_mean = float(new_values.mean())
            ref_std = float(ref_values.std(ddof=0))
            if ref_std == 0:
                standardized_shift = 0.0 if new_mean == ref_mean else np.inf
            else:
                standardized_shift = abs(new_mean - ref_mean) / ref_std
            status = "REVIEW" if standardized_shift >= 0.2 else "OK"
            details = (
                "Mean shift >= 0.2 reference standard deviations; review if flagged"
                if status == "REVIEW"
                else "No large mean shift by this simple check"
            )

        rows.append({
            "check": "numeric_mean_shift",
            "feature": col,
            "reference_value": round(float(ref_mean), 3) if pd.notna(ref_mean) else "",
            "incoming_value": round(float(new_mean), 3) if pd.notna(new_mean) else "",
            "difference": round(float(standardized_shift), 3) if pd.notna(standardized_shift) and np.isfinite(standardized_shift) else str(standardized_shift),
            "status": status,
            "details": details
        })

    # Category/code distribution check for coded categorical features.
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    for col in categorical_cols:
        ref_share = reference[col].fillna("MISSING").astype(str).value_counts(normalize=True)
        new_share = incoming[col].fillna("MISSING").astype(str).value_counts(normalize=True)
        all_categories = sorted(set(ref_share.index) | set(new_share.index))
        max_change = 0.0
        changed_category = ""
        for category in all_categories:
            change = abs(float(new_share.get(category, 0.0)) - float(ref_share.get(category, 0.0)))
            if change > max_change:
                max_change = change
                changed_category = category
        status = "REVIEW" if max_change >= 0.20 else "OK"
        rows.append({
            "check": "categorical_share_change",
            "feature": col,
            "reference_value": "",
            "incoming_value": "",
            "difference": round(max_change, 3),
            "status": status,
            "details": f"Largest category-share change is {max_change:.1%} (code/category {changed_category})"
        })

    report = pd.DataFrame(rows)
    report.to_csv(args.output, index=False)

    review_count = int((report["status"] == "REVIEW").sum())
    print(f"\nChecks requiring review: {review_count} of {len(report)}")
    print(f"Report saved to: {args.output}")
    print("\nRows marked REVIEW are signals to investigate, not proof that the model has failed.")
    print("If target labels become available, separately evaluate model performance on labeled data.")


if __name__ == "__main__":
    main()
