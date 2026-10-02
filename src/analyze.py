"""Reproducible EDA for the Kaggle Titanic training CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def run(data_path: Path, out_dir: Path) -> dict:
    df = pd.read_csv(data_path)
    required = {"Survived", "Pclass", "Sex", "Age", "Fare"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette=["#9D2638", "#1C4058", "#D6A85F"])

    summary = {
        "source_file": data_path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "survival_rate": float(df["Survived"].mean()),
        "survival_by_sex": df.groupby("Sex", dropna=False)["Survived"].agg(["count", "mean"]).to_dict("index"),
        "survival_by_class": df.groupby("Pclass", dropna=False)["Survived"].agg(["count", "mean"]).to_dict("index"),
        "numeric_describe": df.select_dtypes(include="number").describe().to_dict(),
    }

    # Welch t-test: age distributions of recorded survivors and non-survivors.
    age_survived = df.loc[(df.Survived == 1), "Age"].dropna()
    age_not_survived = df.loc[(df.Survived == 0), "Age"].dropna()
    t_stat, t_p = ttest_ind(age_survived, age_not_survived, equal_var=False)
    summary["welch_age_test"] = {
        "group_1": "Survived == 1", "n_1": len(age_survived), "mean_1": age_survived.mean(),
        "group_0": "Survived == 0", "n_0": len(age_not_survived), "mean_0": age_not_survived.mean(),
        "t_statistic": t_stat, "p_value": t_p,
    }

    # Chi-square test: passenger class and survival independence.
    table = pd.crosstab(df["Pclass"], df["Survived"])
    chi2, chi_p, dof, expected = chi2_contingency(table)
    summary["class_survival_chi_square"] = {
        "null": "Passenger class and recorded survival are independent",
        "observed_table": table.to_dict(), "chi2": chi2, "degrees_of_freedom": dof,
        "p_value": chi_p, "minimum_expected_count": expected.min(),
    }

    # Charts use raw observations and show denominator-aware rates where practical.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    rates = df.groupby("Sex")["Survived"].agg(["mean", "count"]).reset_index()
    bars = ax.bar(rates.Sex, rates["mean"], color=["#9D2638", "#1C4058"], width=.58)
    ax.set(title="Recorded survival rate by sex", ylabel="Share recorded as survived", ylim=(0, 1))
    for bar, row in zip(bars, rates.itertuples()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.035,
                f"{row.mean:.0%}  (n={row.count})", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "survival_by_sex.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    rates = df.groupby("Pclass")["Survived"].agg(["mean", "count"]).reset_index()
    bars = ax.bar(rates.Pclass.astype(str), rates["mean"], color=["#D6A85F", "#1C4058", "#9D2638"], width=.58)
    ax.set(title="Recorded survival rate by passenger class", xlabel="Passenger class", ylabel="Share recorded as survived", ylim=(0, 1))
    for bar, row in zip(bars, rates.itertuples()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.035,
                f"{row.mean:.0%}  (n={row.count})", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "survival_by_class.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.histplot(data=df, x="Age", hue="Survived", bins=24, stat="density", common_norm=False,
                 element="step", fill=False, palette={0: "#9D2638", 1: "#1C4058"}, ax=ax)
    ax.set(title="Age distribution by recorded outcome", xlabel="Age (years)", ylabel="Density")
    fig.tight_layout(); fig.savefig(out_dir / "age_by_outcome.png", dpi=180); plt.close(fig)

    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Numeric variable correlations")
    fig.tight_layout(); fig.savefig(out_dir / "correlation_matrix.png", dpi=180); plt.close(fig)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, ensure_ascii=False)
    print(f"Analyzed {len(df)} rows. Results written to {out_dir.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}. Download train.csv from Kaggle; see README.md.")
    run(args.data, args.out)


if __name__ == "__main__":
    main()
