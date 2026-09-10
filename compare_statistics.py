import pandas as pd
import numpy as np
from scipy import stats
import os

# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_DIR = "comparison_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = {
    "association_naming": (
        "Auswertung_Vergleich/Vergleich/stats_assoc_naming.csv",
        "Auswertung_Vergleich/Vergleich/stats_assoc_naming_rep.csv"
    ),
    "association_usage": (
        "Auswertung_Vergleich/Vergleich/stats_assoc_usage.csv",
        "Auswertung_Vergleich/Vergleich/stats_assoc_usage_rep.csv"
    ),
    "decision_naming": (
        "Auswertung_Vergleich/Vergleich/stats_dec_naming.csv",
        "Auswertung_Vergleich/Vergleich/stats_dec_naming_rep.csv"
    ),
    "decision_usage": (
        "Auswertung_Vergleich/Vergleich/stats_dec_usage.csv",
        "Auswertung_Vergleich/Vergleich/stats_dec_usage_rep.csv"
    )
}

COMP_FILES = {
    "association": (
        "Auswertung_Vergleich/Vergleich/comp_assoc.csv",
        "Auswertung_Vergleich/Vergleich/comp_assoc_rep.csv"
    ),
    "decision": (
        "Auswertung_Vergleich/Vergleich/comp_dec.csv",
        "Auswertung_Vergleich/Vergleich/comp_dec_rep.csv"
    )
}

DIALECT_FILES = (
    "Auswertung_Vergleich/Vergleich/dialect_stats.csv",
    "Auswertung_Vergleich/Vergleich/dialect_stats_rep.csv"
)

REJECTION_FILES = (
    "Auswertung_Vergleich/Vergleich/rejection_rates.csv",
    "Auswertung_Vergleich/Vergleich/rejection_rates_rep.csv"
)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def safe_pearson(x, y):
    if len(x) < 2:
        return np.nan, np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    return stats.pearsonr(x, y)


def safe_spearman(x, y):
    if len(x) < 2:
        return np.nan, np.nan

    return stats.spearmanr(x, y)


def comparison_metrics(original, replication):

    original = np.asarray(original, dtype=float)
    replication = np.asarray(replication, dtype=float)

    diff = replication - original
    abs_diff = np.abs(diff)

    pearson_r, pearson_p = safe_pearson(original, replication)
    spearman_rho, spearman_p = safe_spearman(original, replication)

    return {
        "comparisons": len(original),

        "pearson_r": pearson_r,
        "pearson_p": pearson_p,

        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,

        "MAE": np.mean(abs_diff),
        "RMSE": np.sqrt(np.mean(diff ** 2)),
        "max_absolute_difference": np.max(abs_diff),

        "mean_difference_rep_minus_original": np.mean(diff),
        "median_difference_rep_minus_original": np.median(diff),

        "same_sign": np.sum(np.sign(original) == np.sign(replication)),
        "same_sign_percent": np.mean(
            np.sign(original) == np.sign(replication)
        ) * 100
    }


# ============================================================
# 3. COMPARE stats_* FILES
# ============================================================

def compare_stats(original_file, replication_file, experiment):

    original = pd.read_csv(original_file)
    replication = pd.read_csv(replication_file)

    # --------------------------------------------------------
    # Match model × dimension
    # --------------------------------------------------------

    keys = ["model", "dimension"]

    merged = pd.merge(
        original,
        replication,
        on=keys,
        suffixes=("_original", "_replication"),
        how="inner"
    )

    print("\n" + "=" * 80)
    print(experiment)
    print("=" * 80)

    print(
        f"Matched model × dimension combinations: {len(merged)}"
    )

    # --------------------------------------------------------
    # Variables to compare
    # --------------------------------------------------------

    variables = [
        "mean",
        "std",
        "n",
        "t_stat",
        "p_value",
        "cohen_d",
        "ci_low",
        "ci_high"
    ]

    results = []

    for variable in variables:

        col_original = f"{variable}_original"
        col_replication = f"{variable}_replication"

        if col_original not in merged.columns:
            continue

        x = merged[col_original]
        y = merged[col_replication]

        valid = x.notna() & y.notna()

        x = x[valid].values
        y = y[valid].values

        metrics = comparison_metrics(x, y)

        results.append({
            "experiment": experiment,
            "variable": variable,
            **metrics
        })

    comparison = pd.DataFrame(results)

    # --------------------------------------------------------
    # Significance agreement
    # --------------------------------------------------------

    if (
        "significant_original" in merged.columns
        and "significant_replication" in merged.columns
    ):

        sig_original = merged["significant_original"].astype(bool)
        sig_replication = merged["significant_replication"].astype(bool)

        same_significance = sig_original == sig_replication

        sig_agreement = pd.DataFrame([{
            "experiment": experiment,

            "comparisons": len(merged),

            "original_significant": sig_original.sum(),
            "replication_significant": sig_replication.sum(),

            "same_significance": same_significance.sum(),

            "same_significance_percent":
                same_significance.mean() * 100,

            "original_yes_rep_no":
                ((sig_original == True) &
                 (sig_replication == False)).sum(),

            "original_no_rep_yes":
                ((sig_original == False) &
                 (sig_replication == True)).sum()
        }])

    else:
        sig_agreement = pd.DataFrame()

    # --------------------------------------------------------
    # Detailed comparison
    # --------------------------------------------------------

    detail = merged[
        keys +
        [
            "mean_original",
            "mean_replication",
            "std_original",
            "std_replication",
            "n_original",
            "n_replication",
            "t_stat_original",
            "t_stat_replication",
            "p_value_original",
            "p_value_replication",
            "cohen_d_original",
            "cohen_d_replication",
            "ci_low_original",
            "ci_low_replication",
            "ci_high_original",
            "ci_high_replication",
            "significant_original",
            "significant_replication"
        ]
    ].copy()

    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    detail["mean_difference"] = (
        detail["mean_replication"]
        - detail["mean_original"]
    )

    detail["cohen_d_difference"] = (
        detail["cohen_d_replication"]
        - detail["cohen_d_original"]
    )

    detail["ci_width_original"] = (
        detail["ci_high_original"]
        - detail["ci_low_original"]
    )

    detail["ci_width_replication"] = (
        detail["ci_high_replication"]
        - detail["ci_low_replication"]
    )

    detail["significance_agrees"] = (
        detail["significant_original"]
        == detail["significant_replication"]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    comparison.to_csv(
        f"{OUTPUT_DIR}/{experiment}_statistics_comparison.csv",
        index=False
    )

    detail.to_csv(
        f"{OUTPUT_DIR}/{experiment}_statistics_detail.csv",
        index=False
    )

    if not sig_agreement.empty:
        sig_agreement.to_csv(
            f"{OUTPUT_DIR}/{experiment}_significance_agreement.csv",
            index=False
        )

    return comparison, detail, sig_agreement


# ============================================================
# 4. RUN stats_* COMPARISONS
# ============================================================

all_stats_comparisons = []
all_sig_agreements = []

for experiment, (original_file, replication_file) in EXPERIMENTS.items():

    comparison, detail, sig_agreement = compare_stats(
        original_file,
        replication_file,
        experiment
    )

    all_stats_comparisons.append(comparison)

    if not sig_agreement.empty:
        all_sig_agreements.append(sig_agreement)


all_stats_comparisons = pd.concat(
    all_stats_comparisons,
    ignore_index=True
)

all_sig_agreements = pd.concat(
    all_sig_agreements,
    ignore_index=True
)


# ============================================================
# 5. PRINT MAIN RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("COMPARISON OF STATISTICAL RESULTS")
print("=" * 100)

print(
    all_stats_comparisons[
        [
            "experiment",
            "variable",
            "comparisons",
            "pearson_r",
            "spearman_rho",
            "MAE",
            "RMSE",
            "max_absolute_difference",
            "mean_difference_rep_minus_original",
            "same_sign_percent"
        ]
    ].round(4).to_string(index=False)
)


print("\n")
print("=" * 100)
print("SIGNIFICANCE AGREEMENT")
print("=" * 100)

print(
    all_sig_agreements.round(4).to_string(index=False)
)


# ============================================================
# 6. SPECIAL FOCUS: MEAN BIAS
# ============================================================

mean_results = all_stats_comparisons[
    all_stats_comparisons["variable"] == "mean"
].copy()

mean_results.to_csv(
    f"{OUTPUT_DIR}/ALL_MEAN_COMPARISONS.csv",
    index=False
)

print("\n")
print("=" * 100)
print("MEAN BIAS COMPARISON")
print("=" * 100)

print(
    mean_results[
        [
            "experiment",
            "comparisons",
            "pearson_r",
            "spearman_rho",
            "MAE",
            "RMSE",
            "max_absolute_difference",
            "mean_difference_rep_minus_original",
            "same_sign_percent"
        ]
    ].round(4).to_string(index=False)
)


# ============================================================
# 7. COMPARE Naming vs Usage ANALYSIS
# ============================================================

def compare_comp_files(original_file, replication_file, task):

    original = pd.read_csv(original_file)
    replication = pd.read_csv(replication_file)

    keys = ["model", "dimension"]

    merged = pd.merge(
        original,
        replication,
        on=keys,
        suffixes=("_original", "_replication"),
        how="inner"
    )

    print("\n")
    print("=" * 100)
    print(f"Naming vs Usage comparison: {task}")
    print("=" * 100)

    results = []

    # Compare:
    # mean_naming
    # mean_usage
    # t_stat
    # p_value

    for variable in [
        "mean_naming",
        "mean_usage",
        "t_stat",
        "p_value"
    ]:

        x = merged[f"{variable}_original"]
        y = merged[f"{variable}_replication"]

        valid = x.notna() & y.notna()

        x = x[valid].values
        y = y[valid].values

        metrics = comparison_metrics(x, y)

        results.append({
            "task": task,
            "variable": variable,
            **metrics
        })

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Compare naming_higher
    # --------------------------------------------------------

    if (
        "naming_higher_original" in merged.columns
        and "naming_higher_replication" in merged.columns
    ):

        same_direction = (
            merged["naming_higher_original"]
            == merged["naming_higher_replication"]
        )

        direction_result = pd.DataFrame([{
            "task": task,
            "comparisons": len(merged),

            "original_naming_higher":
                merged["naming_higher_original"].sum(),

            "replication_naming_higher":
                merged["naming_higher_replication"].sum(),

            "same_direction":
                same_direction.sum(),

            "same_direction_percent":
                same_direction.mean() * 100
        }])

        direction_result.to_csv(
            f"{OUTPUT_DIR}/{task}_naming_usage_direction.csv",
            index=False
        )

        print("\nNaming > Usage agreement:")
        print(direction_result.to_string(index=False))

    result_df.to_csv(
        f"{OUTPUT_DIR}/{task}_naming_usage_comparison.csv",
        index=False
    )

    return result_df


all_comp = []

for task, (original_file, replication_file) in COMP_FILES.items():

    result = compare_comp_files(
        original_file,
        replication_file,
        task
    )

    all_comp.append(result)

all_comp = pd.concat(
    all_comp,
    ignore_index=True
)


# ============================================================
# 8. DIALECT STATS
# ============================================================

def compare_dialect_stats(original_file, replication_file):

    original = pd.read_csv(original_file)
    replication = pd.read_csv(replication_file)

    keys = ["language", "dimension"]

    merged = pd.merge(
        original,
        replication,
        on=keys,
        suffixes=("_original", "_replication"),
        how="inner"
    )

    x = merged["mean_original"].values
    y = merged["mean_replication"].values

    metrics = comparison_metrics(x, y)

    result = pd.DataFrame([{
        "analysis": "dialect_stats",
        **metrics
    }])

    # Detailed differences
    merged["mean_difference"] = (
        merged["mean_replication"]
        - merged["mean_original"]
    )

    merged["ci_low_difference"] = (
        merged["ci_low_replication"]
        - merged["ci_low_original"]
    )

    merged["ci_high_difference"] = (
        merged["ci_high_replication"]
        - merged["ci_high_original"]
    )

    result.to_csv(
        f"{OUTPUT_DIR}/dialect_stats_comparison.csv",
        index=False
    )

    merged.to_csv(
        f"{OUTPUT_DIR}/dialect_stats_detail.csv",
        index=False
    )

    print("\n")
    print("=" * 100)
    print("DIALECT STATS COMPARISON")
    print("=" * 100)

    print(result.round(4).to_string(index=False))

    return result


dialect_comparison = compare_dialect_stats(
    DIALECT_FILES[0],
    DIALECT_FILES[1]
)


# ============================================================
# 9. REJECTION RATES
# ============================================================

def compare_rejection_rates(original_file, replication_file):

    original = pd.read_csv(original_file)
    replication = pd.read_csv(replication_file)

    merged = pd.merge(
        original,
        replication,
        on="model_name",
        suffixes=("_original", "_replication"),
        how="inner"
    )

    x = merged["rejected_original"].values
    y = merged["rejected_replication"].values

    metrics = comparison_metrics(x, y)

    result = pd.DataFrame([{
        "analysis": "rejection_rates",
        **metrics
    }])

    merged["difference"] = (
        merged["rejected_replication"]
        - merged["rejected_original"]
    )

    result.to_csv(
        f"{OUTPUT_DIR}/rejection_rates_comparison.csv",
        index=False
    )

    merged.to_csv(
        f"{OUTPUT_DIR}/rejection_rates_detail.csv",
        index=False
    )

    print("\n")
    print("=" * 100)
    print("REJECTION RATE COMPARISON")
    print("=" * 100)

    print(result.round(4).to_string(index=False))

    return result


rejection_comparison = compare_rejection_rates(
    REJECTION_FILES[0],
    REJECTION_FILES[1]
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

final_summary = mean_results[
    [
        "experiment",
        "comparisons",
        "pearson_r",
        "pearson_p",
        "spearman_rho",
        "spearman_p",
        "MAE",
        "RMSE",
        "max_absolute_difference",
        "mean_difference_rep_minus_original",
        "median_difference_rep_minus_original",
        "same_sign_percent"
    ]
].copy()

final_summary.to_csv(
    f"{OUTPUT_DIR}/FINAL_SUMMARY.csv",
    index=False
)

print("\n")
print("=" * 100)
print("FINAL SUMMARY – MEAN BIAS")
print("=" * 100)

print(
    final_summary.round(4).to_string(index=False)
)


print("\n")
print("=" * 100)
print("FILES SAVED")
print("=" * 100)

print(f"All comparison files saved to: {OUTPUT_DIR}/")