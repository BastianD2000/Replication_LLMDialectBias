import pandas as pd
import numpy as np
from scipy import stats
import os


# 1. CONFIGURATION

OUTPUT_DIR = "comparison_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)



# Main Bias Score comparisons


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



# Naming vs. Usage comparisons


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



# Dialect comparisons


DIALECT_FILES = {
    "association_naming": (
        "Auswertung_Vergleich/Vergleich/dialect_stats_assoc_naming.csv",
        "Auswertung_Vergleich/Vergleich/dialect_stats_assoc_naming_rep.csv"
    ),
    "association_usage": (
        "Auswertung_Vergleich/Vergleich/dialect_stats_assoc_usage.csv",
        "Auswertung_Vergleich/Vergleich/dialect_stats_assoc_usage_rep.csv"
    ),
    "decision_naming": (
        "Auswertung_Vergleich/Vergleich/dialect_stats_dec_naming.csv",
        "Auswertung_Vergleich/Vergleich/dialect_stats_dec_naming_rep.csv"
    ),
    "decision_usage": (
        "Auswertung_Vergleich/Vergleich/dialect_stats_dec_usage.csv",
        "Auswertung_Vergleich/Vergleich/dialect_stats_dec_usage_rep.csv"
    )
}



# Rejection rates


REJECTION_FILES = (
    "Auswertung_Vergleich/Vergleich/rejection_rates.csv",
    "Auswertung_Vergleich/Vergleich/rejection_rates_rep.csv"
)



# 2. HELPER FUNCTION


def comparison_metrics(original, replication):
    """
    Compares two sets of numerical results.

    Metrics:
    - Pearson correlation
    - Spearman correlation
    - MAE
    - RMSE
    - maximum absolute difference
    - mean difference
    - percentage with same sign
    """

    original = np.asarray(original, dtype=float)
    replication = np.asarray(replication, dtype=float)

    # Remove missing values
    valid = np.isfinite(original) & np.isfinite(replication)

    original = original[valid]
    replication = replication[valid]

    if len(original) == 0:
        return {
            "comparisons": 0,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "max_absolute_difference": np.nan,
            "mean_difference_rep_minus_original": np.nan,
            "same_sign_percent": np.nan
        }

    # Differences
    diff = replication - original
    abs_diff = np.abs(diff)

  
    # Pearson
    

    if len(original) >= 2 and np.std(original) > 0 and np.std(replication) > 0:
        pearson_r = stats.pearsonr(
            original,
            replication
        ).statistic
    else:
        pearson_r = np.nan

    
    # Spearman
    

    if len(original) >= 2:
        spearman_rho = stats.spearmanr(
            original,
            replication
        ).statistic
    else:
        spearman_rho = np.nan

    
    # Return relevant metrics
    

    return {
        "comparisons": len(original),

        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,

        "MAE": np.mean(abs_diff),

        "RMSE": np.sqrt(
            np.mean(diff ** 2)
        ),

        "max_absolute_difference": np.max(abs_diff),

        "mean_difference_rep_minus_original":
            np.mean(diff),

        "same_sign_percent":
            np.mean(
                np.sign(original) == np.sign(replication)
            ) * 100
    }



# 3. MAIN BIAS SCORE COMPARISON


def compare_bias_scores(
    original_file,
    replication_file,
    experiment
):

    original = pd.read_csv(original_file)
    replication = pd.read_csv(replication_file)

    
    # Match model × dimension
    

    keys = ["model", "dimension"]

    merged = pd.merge(
        original,
        replication,
        on=keys,
        suffixes=("_original", "_replication"),
        how="inner"
    )

    print("\n" + "=" * 80)
    print(f"BIAS SCORE COMPARISON: {experiment}")
    print("=" * 80)

    print(
        f"Matched model × dimension combinations: "
        f"{len(merged)}"
    )

    
    # Compare Bias Scores
    

    x = merged["mean_original"].values
    y = merged["mean_replication"].values

    metrics = comparison_metrics(x, y)

    result = pd.DataFrame([{
        "experiment": experiment,
        **metrics
    }])

    
    # Detailed Bias Score comparison
    

    detail = merged[
        keys +
        [
            "mean_original",
            "mean_replication"
        ]
    ].copy()

    detail["difference_rep_minus_original"] = (
        detail["mean_replication"]
        - detail["mean_original"]
    )

    detail["absolute_difference"] = (
        np.abs(
            detail["difference_rep_minus_original"]
        )
    )

    detail["same_sign"] = (
        np.sign(detail["mean_original"])
        ==
        np.sign(detail["mean_replication"])
    )

    
    # Save results
    

    result.to_csv(
        f"{OUTPUT_DIR}/{experiment}_bias_score_comparison.csv",
        index=False
    )

    detail.to_csv(
        f"{OUTPUT_DIR}/{experiment}_bias_score_detail.csv",
        index=False
    )

    print("\nBias Score comparison:")
    print(
        result.round(4).to_string(index=False)
    )

    return result, detail



# 4. SIGNIFICANCE AGREEMENT


def compare_significance(
    original_file,
    replication_file,
    experiment
):

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

    if (
        "significant_original" not in merged.columns
        or
        "significant_replication" not in merged.columns
    ):
        print(
            f"WARNING: No significance columns found "
            f"for {experiment}"
        )

        return pd.DataFrame(), pd.DataFrame()

    
    # Convert to boolean
    

    sig_original = (
        merged["significant_original"]
        .astype(bool)
    )

    sig_replication = (
        merged["significant_replication"]
        .astype(bool)
    )

    same_significance = (
        sig_original == sig_replication
    )

    
    # Summary
    

    result = pd.DataFrame([{
        "experiment": experiment,

        "comparisons": len(merged),

        "original_significant":
            sig_original.sum(),

        "replication_significant":
            sig_replication.sum(),

        "same_significance":
            same_significance.sum(),

        "same_significance_percent":
            same_significance.mean() * 100,

        "original_yes_rep_no":
            (
                sig_original
                &
                ~sig_replication
            ).sum(),

        "original_no_rep_yes":
            (
                ~sig_original
                &
                sig_replication
            ).sum()
    }])

    
    # Detailed comparison
    

    detail = merged[
        keys +
        [
            "significant_original",
            "significant_replication"
        ]
    ].copy()

    detail["significance_agrees"] = (
        detail["significant_original"]
        ==
        detail["significant_replication"]
    )

    
    # Save
    

    result.to_csv(
        f"{OUTPUT_DIR}/{experiment}_significance_agreement.csv",
        index=False
    )

    detail.to_csv(
        f"{OUTPUT_DIR}/{experiment}_significance_detail.csv",
        index=False
    )

    print("\nSignificance agreement:")
    print(
        result.round(4).to_string(index=False)
    )

    return result, detail



# 5. RUN MAIN COMPARISONS


all_bias_results = []
all_significance_results = []

for experiment, (
    original_file,
    replication_file
) in EXPERIMENTS.items():

    # Bias Scores
    bias_result, bias_detail = compare_bias_scores(
        original_file,
        replication_file,
        experiment
    )

    all_bias_results.append(
        bias_result
    )

    # Significance
    significance_result, significance_detail = (
        compare_significance(
            original_file,
            replication_file,
            experiment
        )
    )

    if not significance_result.empty:
        all_significance_results.append(
            significance_result
        )


all_bias_results = pd.concat(
    all_bias_results,
    ignore_index=True
)

all_significance_results = pd.concat(
    all_significance_results,
    ignore_index=True
)



# 6. MAIN SUMMARY


all_bias_results.to_csv(
    f"{OUTPUT_DIR}/ALL_BIAS_SCORE_COMPARISONS.csv",
    index=False
)

all_significance_results.to_csv(
    f"{OUTPUT_DIR}/ALL_SIGNIFICANCE_COMPARISONS.csv",
    index=False
)


print("\n")
print("=" * 100)
print("MAIN BIAS SCORE COMPARISON")
print("=" * 100)

print(
    all_bias_results[
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
    ]
    .round(4)
    .to_string(index=False)
)


print("\n")
print("=" * 100)
print("SIGNIFICANCE AGREEMENT")
print("=" * 100)

print(
    all_significance_results
    .round(4)
    .to_string(index=False)
)



# 7. NAMING VS. USAGE


def compare_naming_usage(
    original_file,
    replication_file,
    task
):

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
    print(f"NAMING VS. USAGE: {task}")
    print("=" * 100)

    
    # Mean Naming
    

    naming_metrics = comparison_metrics(
        merged["mean_naming_original"],
        merged["mean_naming_replication"]
    )

    
    # Mean Usage
    

    usage_metrics = comparison_metrics(
        merged["mean_usage_original"],
        merged["mean_usage_replication"]
    )

    result = pd.DataFrame([
        {
            "task": task,
            "variable": "mean_naming",
            **naming_metrics
        },
        {
            "task": task,
            "variable": "mean_usage",
            **usage_metrics
        }
    ])

    
    # Direction Naming > Usage
   

    if (
        "naming_higher_original" in merged.columns
        and
        "naming_higher_replication" in merged.columns
    ):

        same_direction = (
            merged["naming_higher_original"]
            ==
            merged["naming_higher_replication"]
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
        print(
            direction_result
            .round(4)
            .to_string(index=False)
        )

    else:
        direction_result = pd.DataFrame()

    
    # Save
    

    result.to_csv(
        f"{OUTPUT_DIR}/{task}_naming_usage_comparison.csv",
        index=False
    )

    print("\nNaming / Usage comparison:")
    print(
        result.round(4).to_string(index=False)
    )

    return result, direction_result


all_naming_usage_results = []
all_direction_results = []

for task, (
    original_file,
    replication_file
) in COMP_FILES.items():

    result, direction_result = compare_naming_usage(
        original_file,
        replication_file,
        task
    )

    all_naming_usage_results.append(result)

    if not direction_result.empty:
        all_direction_results.append(
            direction_result
        )


all_naming_usage_results = pd.concat(
    all_naming_usage_results,
    ignore_index=True
)

all_naming_usage_results.to_csv(
    f"{OUTPUT_DIR}/ALL_NAMING_USAGE_COMPARISONS.csv",
    index=False
)

if all_direction_results:
    all_direction_results = pd.concat(
        all_direction_results,
        ignore_index=True
    )

    all_direction_results.to_csv(
        f"{OUTPUT_DIR}/ALL_NAMING_USAGE_DIRECTION.csv",
        index=False
    )



# 8. DIALECT COMPARISON


def compare_dialect_stats(
    original_file,
    replication_file,
    experiment
):

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

    print("\n")
    print("=" * 100)
    print(f"DIALECT BIAS SCORE COMPARISON: {experiment}")
    print("=" * 100)

    print(
        f"Matched language × dimension combinations: "
        f"{len(merged)}"
    )

    
    # Compare dialect-level Bias Scores
    

    metrics = comparison_metrics(
        merged["mean_original"].values,
        merged["mean_replication"].values
    )

    result = pd.DataFrame([{
        "experiment": experiment,
        **metrics
    }])

    
    # Detailed comparison
    

    detail = merged[
        keys +
        [
            "mean_original",
            "mean_replication"
        ]
    ].copy()

    detail["difference_rep_minus_original"] = (
        detail["mean_replication"]
        - detail["mean_original"]
    )

    detail["absolute_difference"] = (
        np.abs(
            detail["difference_rep_minus_original"]
        )
    )

    detail["same_sign"] = (
        np.sign(detail["mean_original"])
        ==
        np.sign(detail["mean_replication"])
    )

    
    # Save
    

    result.to_csv(
        f"{OUTPUT_DIR}/{experiment}_dialect_comparison.csv",
        index=False
    )

    detail.to_csv(
        f"{OUTPUT_DIR}/{experiment}_dialect_detail.csv",
        index=False
    )

    print("\nDialect comparison:")
    print(
        result.round(4).to_string(index=False)
    )

    return result, detail


all_dialect_results = []

for experiment, (
    original_file,
    replication_file
) in DIALECT_FILES.items():

    result, detail = compare_dialect_stats(
        original_file,
        replication_file,
        experiment
    )

    all_dialect_results.append(
        result
    )


all_dialect_results = pd.concat(
    all_dialect_results,
    ignore_index=True
)

all_dialect_results.to_csv(
    f"{OUTPUT_DIR}/ALL_DIALECT_COMPARISONS.csv",
    index=False
)



# 9. REJECTION RATES


def compare_rejection_rates(
    original_file,
    replication_file
):

    original = pd.read_csv(original_file)
    replication = pd.read_csv(replication_file)

    merged = pd.merge(
        original,
        replication,
        on="model_name",
        suffixes=("_original", "_replication"),
        how="inner"
    )

    
    # Direct comparison
    

    merged["difference_rep_minus_original"] = (
        merged["rejected_replication"]
        -
        merged["rejected_original"]
    )

    merged["absolute_difference"] = (
        np.abs(
            merged["difference_rep_minus_original"]
        )
    )

    
    # Save detailed comparison
    

    merged.to_csv(
        f"{OUTPUT_DIR}/rejection_rates_detail.csv",
        index=False
    )

    print("\n")
    print("=" * 100)
    print("REJECTION RATE COMPARISON")
    print("=" * 100)

    print(
        merged[
            [
                "model_name",
                "rejected_original",
                "rejected_replication",
                "difference_rep_minus_original"
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    return merged


rejection_comparison = compare_rejection_rates(
    REJECTION_FILES[0],
    REJECTION_FILES[1]
)



# 10. FINAL SUMMARY


final_summary = all_bias_results[
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
].copy()


final_summary.to_csv(
    f"{OUTPUT_DIR}/FINAL_SUMMARY.csv",
    index=False
)


print("\n")
print("=" * 100)
print("FINAL SUMMARY – BIAS SCORES")
print("=" * 100)

print(
    final_summary
    .round(4)
    .to_string(index=False)
)


print("\n")
print("=" * 100)
print("FILES SAVED")
print("=" * 100)

print(
    f"All comparison files saved to: "
    f"{OUTPUT_DIR}/"
)