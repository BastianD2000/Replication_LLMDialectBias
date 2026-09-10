import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import (
    ttest_ind,
    pearsonr,
    spearmanr
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# ============================================================
# KONFIGURATION
# ============================================================

# Hier deine Dateien eintragen
FILES = {
    "association_naming": {
        "original": "Auswertung_Vergleich/Vergleich_Rohwerte/output_original/implicit_explicit/eval/final.csv",
        "replication": "Auswertung_Vergleich/Vergleich_Rohwerte/output_rep/association_naming/eval/final.csv",
    },

    "association_usage": {
        "original": "Auswertung_Vergleich/Vergleich_Rohwerte/output_original/implicit/eval/final.csv",
        "replication": "Auswertung_Vergleich/Vergleich_Rohwerte/output_rep/association_usage/eval/final.csv",
    },

    "decision_naming": {
        "original": "Auswertung_Vergleich/Vergleich_Rohwerte/output_original/decision_explicit_extracted/eval/final.csv",
        "replication": "Auswertung_Vergleich/Vergleich_Rohwerte/output_rep/decision_extracted_namingeval/final.csv",
    },

    "decision_usage": {
        "original": "Auswertung_Vergleich/Vergleich_Rohwerte/output_original/decision_extracted/eval/final.csv",
        "replication": "Auswertung_Vergleich/Vergleich_Rohwerte/output_rep/decision_extracted_usageeval/final.csv",
    }
}

OUTPUT_DIR = Path("replication_statistics")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def cohens_d(x, y):
    """
    Cohen's d für zwei unabhängige Gruppen.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx = len(x)
    ny = len(y)

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled_sd = np.sqrt(
        ((nx - 1) * vx + (ny - 1) * vy)
        / (nx + ny - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (np.mean(y) - np.mean(x)) / pooled_sd


def mean_ci_difference(x, y, confidence=0.95):
    """
    95%-KI für Mittelwertsdifferenz:
    replication - original

    Welch-Standardfehler.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nx = len(x)
    ny = len(y)

    mean_diff = np.mean(y) - np.mean(x)

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    se = np.sqrt(vx / nx + vy / ny)

    if se == 0:
        return mean_diff, np.nan, np.nan

    # Welch degrees of freedom
    df = (
        (vx / nx + vy / ny) ** 2
        /
        (
            (vx / nx) ** 2 / (nx - 1)
            +
            (vy / ny) ** 2 / (ny - 1)
        )
    )

    from scipy.stats import t

    alpha = 1 - confidence
    critical = t.ppf(1 - alpha / 2, df)

    margin = critical * se

    return (
        mean_diff,
        mean_diff - margin,
        mean_diff + margin
    )


# ============================================================
# DATEN EINLESEN
# ============================================================

def load_data(path):

    df = pd.read_csv(path)

    required = [
        "bias",
        "model_name",
        "dimension",
        "language"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{path} enthält nicht die benötigten Spalten: {missing}"
        )

    df["bias"] = pd.to_numeric(
        df["bias"],
        errors="coerce"
    )

    df = df.dropna(subset=["bias"])

    return df


# ============================================================
# VERGLEICH EINER EXPERIMENTVARIANTE
# ============================================================

def compare_experiment(name, original_path, replication_path):

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    original = load_data(original_path)
    replication = load_data(replication_path)

    print(f"Original Beobachtungen:     {len(original)}")
    print(f"Replikation Beobachtungen:  {len(replication)}")

    # --------------------------------------------------------
    # GEMEINSAME MODELLE
    # --------------------------------------------------------

    original_models = set(original["model_name"].unique())
    replication_models = set(replication["model_name"].unique())

    common_models = sorted(
        original_models.intersection(replication_models)
    )

    only_original = sorted(
        original_models - replication_models
    )

    only_replication = sorted(
        replication_models - original_models
    )

    print("\nModelle im Original:")
    print(original_models)

    print("\nModelle in der Replikation:")
    print(replication_models)

    print("\nGemeinsame Modelle:")
    print(common_models)

    if only_original:
        print("\nNur im Original:")
        print(only_original)

    if only_replication:
        print("\nNur in der Replikation:")
        print(only_replication)

    # Nur gemeinsame Modelle verwenden
    original = original[
        original["model_name"].isin(common_models)
    ].copy()

    replication = replication[
        replication["model_name"].isin(common_models)
    ].copy()

    # --------------------------------------------------------
    # GEMEINSAME DIMENSIONEN
    # --------------------------------------------------------

    original_dims = set(original["dimension"].unique())
    replication_dims = set(replication["dimension"].unique())

    common_dims = sorted(
        original_dims.intersection(replication_dims)
    )

    print("\nGemeinsame Dimensionen:")
    print(common_dims)

    # --------------------------------------------------------
    # VERGLEICH MODELL × DIMENSION
    # --------------------------------------------------------

    results = []

    for model in common_models:

        for dimension in common_dims:

            orig_group = original[
                (original["model_name"] == model)
                &
                (original["dimension"] == dimension)
            ]["bias"].dropna()

            rep_group = replication[
                (replication["model_name"] == model)
                &
                (replication["dimension"] == dimension)
            ]["bias"].dropna()

            if len(orig_group) == 0 or len(rep_group) == 0:
                continue

            # Mittelwerte
            mean_orig = orig_group.mean()
            mean_rep = rep_group.mean()

            difference = mean_rep - mean_orig

            # Welch t-test
            t_stat, p_value = ttest_ind(
                orig_group,
                rep_group,
                equal_var=False
            )

            # Effektgröße
            d = cohens_d(
                orig_group,
                rep_group
            )

            # KI
            diff, ci_low, ci_high = mean_ci_difference(
                orig_group,
                rep_group
            )

            # Vorzeichen
            sign_orig = np.sign(mean_orig)
            sign_rep = np.sign(mean_rep)

            sign_match = (
                sign_orig == sign_rep
            )

            results.append({
                "model": model,
                "dimension": dimension,

                "mean_original": mean_orig,
                "std_original": orig_group.std(ddof=1),
                "n_original": len(orig_group),

                "mean_replication": mean_rep,
                "std_replication": rep_group.std(ddof=1),
                "n_replication": len(rep_group),

                "difference_rep_minus_original": difference,
                "absolute_difference": abs(difference),

                "t_stat": t_stat,
                "p_value": p_value,

                "cohens_d": d,

                "ci_low": ci_low,
                "ci_high": ci_high,

                "sign_original": sign_orig,
                "sign_replication": sign_rep,
                "sign_match": sign_match
            })

    results = pd.DataFrame(results)

    # --------------------------------------------------------
    # SIGNIFIKANZ
    # --------------------------------------------------------

    results["significant_difference"] = (
        results["p_value"] < 0.05
    )

    # --------------------------------------------------------
    # GESAMTÜBEREINSTIMMUNG
    # --------------------------------------------------------

    original_means = results["mean_original"]
    replication_means = results["mean_replication"]

    pearson_r, pearson_p = pearsonr(
        original_means,
        replication_means
    )

    spearman_rho, spearman_p = spearmanr(
        original_means,
        replication_means
    )

    mae = mean_absolute_error(
        original_means,
        replication_means
    )

    rmse = np.sqrt(
        mean_squared_error(
            original_means,
            replication_means
        )
    )

    max_difference = results[
        "absolute_difference"
    ].max()

    mean_difference = results[
        "difference_rep_minus_original"
    ].mean()

    median_difference = results[
        "difference_rep_minus_original"
    ].median()

    # --------------------------------------------------------
    # SIGNIFIKANZZAHLEN
    # --------------------------------------------------------

    significant_differences = (
        results["significant_difference"].sum()
    )

    sign_matches = (
        results["sign_match"].sum()
    )

    total_comparisons = len(results)

    # --------------------------------------------------------
    # ERGEBNISSE AUSGEBEN
    # --------------------------------------------------------

    print("\n--- Vergleich der Mittelwerte ---")

    print(
        results[
            [
                "model",
                "dimension",
                "mean_original",
                "mean_replication",
                "difference_rep_minus_original",
                "p_value",
                "cohens_d"
            ]
        ].to_string(index=False)
    )

    print("\n--- Gesamtvergleich ---")

    print(
        f"Verglichene Modell × Dimension Kombinationen: "
        f"{total_comparisons}"
    )

    print(
        f"Pearson r:      {pearson_r:.4f}"
    )

    print(
        f"Pearson p:      {pearson_p:.4e}"
    )

    print(
        f"Spearman rho:   {spearman_rho:.4f}"
    )

    print(
        f"Spearman p:     {spearman_p:.4e}"
    )

    print(
        f"MAE:            {mae:.4f}"
    )

    print(
        f"RMSE:           {rmse:.4f}"
    )

    print(
        f"Max. Abweichung:{max_difference:.4f}"
    )

    print(
        f"Mittlere Δ:     {mean_difference:.4f}"
    )

    print(
        f"Median Δ:       {median_difference:.4f}"
    )

    print(
        f"Signifikante Unterschiede: "
        f"{significant_differences}/{total_comparisons}"
    )

    print(
        f"Gleiches Vorzeichen: "
        f"{sign_matches}/{total_comparisons}"
    )

    # --------------------------------------------------------
    # GRÖSSTE ABWEICHUNGEN
    # --------------------------------------------------------

    print("\n--- Größte Abweichungen ---")

    largest = results.sort_values(
        "absolute_difference",
        ascending=False
    ).head(10)

    print(
        largest[
            [
                "model",
                "dimension",
                "mean_original",
                "mean_replication",
                "difference_rep_minus_original",
                "p_value"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # GESAMTSTATISTIK SPEICHERN
    # --------------------------------------------------------

    summary = pd.DataFrame([{
        "experiment": name,

        "common_models": len(common_models),
        "comparisons": total_comparisons,

        "pearson_r": pearson_r,
        "pearson_p": pearson_p,

        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,

        "MAE": mae,
        "RMSE": rmse,
        "max_absolute_difference": max_difference,

        "mean_difference_rep_minus_original":
            mean_difference,

        "median_difference_rep_minus_original":
            median_difference,

        "significant_differences":
            significant_differences,

        "significant_difference_percent":
            significant_differences
            / total_comparisons * 100,

        "same_sign":
            sign_matches,

        "same_sign_percent":
            sign_matches
            / total_comparisons * 100
    }])

    # --------------------------------------------------------
    # SPEICHERN
    # --------------------------------------------------------

    results_file = (
        OUTPUT_DIR
        / f"{name}_comparison.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / f"{name}_summary.csv"
    )

    results.to_csv(
        results_file,
        index=False
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        f"\nGespeichert: {results_file}"
    )

    print(
        f"Gespeichert: {summary_file}"
    )

    return results, summary


# ============================================================
# ALLE EXPERIMENTE AUSFÜHREN
# ============================================================

all_summaries = []

for name, files in FILES.items():

    results, summary = compare_experiment(
        name,
        files["original"],
        files["replication"]
    )

    all_summaries.append(summary)


# ============================================================
# GESAMTÜBERSICHT
# ============================================================

if all_summaries:

    overview = pd.concat(
        all_summaries,
        ignore_index=True
    )

    overview_file = (
        OUTPUT_DIR
        / "ALL_EXPERIMENTS_SUMMARY.csv"
    )

    overview.to_csv(
        overview_file,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("GESAMTÜBERSICHT")
    print("=" * 70)

    print(
        overview[
            [
                "experiment",
                "common_models",
                "comparisons",
                "pearson_r",
                "spearman_rho",
                "MAE",
                "RMSE",
                "max_absolute_difference",
                "significant_differences",
                "significant_difference_percent",
                "same_sign_percent"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nGespeichert: {overview_file}"
    )