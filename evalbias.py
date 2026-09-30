import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import bootstrap
import os


# 1. Configuration


# File paths (adjust as needed)
ASSOC_NAMING = "German-Dialect-Bias-main/output/association_naming/eval/final.csv" 
ASSOC_USAGE = "German-Dialect-Bias-main/output/association_usage/eval/final.csv"
DEC_NAMING = "German-Dialect-Bias-main/output/decision_extracted_namingeval/final.csv"
DEC_USAGE = "German-Dialect-Bias-main/output/decision_extracted_usageeval/final.csv"

# Models to consider (ordered as in paper)
MODELS = [
    "Llama-3.1 8B",
    "Qwen2.5 7B",
    "Aya 32b",
    "Aya 8B",
    "Gemma-3 12B"
]

DIM_MAP = {
    "friendly": "Friendly",
    "educated": "Uneducated",
    "urban": "Rural",
    "calm": "Temper",
    "open_to_experience": "Closed-Minded",
    "conscientiousness": "Careless"
}
DIM_ORDER = ["Careless", "Closed-Minded", "Friendly", "Rural", "Temper", "Uneducated"]


# 2. Helper functions (statistics & plotting)


def preprocess_df(df):
    if 'dimension' in df.columns:
        df['dimension'] = df['dimension'].apply(lambda x: x.split('-')[0] if '-' in str(x) else x)
        df['dimension'] = df['dimension'].replace(DIM_MAP)
    return df

def bootstrap_ci(data, n_resamples=1000, ci=0.95):
    data = (np.array(data),)
    res = bootstrap(data, np.mean, n_resamples=n_resamples, confidence_level=ci, method='percentile')
    return res.confidence_interval.low, res.confidence_interval.high

def compute_stats(df, model, dimension):
    subset = df[(df['model_name'] == model) & (df['dimension'] == dimension)].dropna(subset=['bias'])
    bias_values = subset['bias'].values
    n = len(bias_values)
    if n < 2:
        return None
    mean = np.mean(bias_values)
    std = np.std(bias_values, ddof=1)
    sem = std / np.sqrt(n)
    t_stat, p_val = stats.ttest_1samp(bias_values, 0)
    cohen_d = mean / std if std != 0 else np.nan
    ci_low, ci_high = bootstrap_ci(bias_values)
    return {
        'model': model,
        'dimension': dimension,
        'mean': mean,
        'std': std,
        'n': n,
        't_stat': t_stat,
        'p_value': p_val,
        'cohen_d': cohen_d,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'significant': p_val < 0.001
    }

def compute_all_stats(df):
    results = []
    for model in MODELS:
        for dim in DIM_ORDER:
            stat = compute_stats(df, model, dim)
            if stat is not None:
                results.append(stat)
    return pd.DataFrame(results)


# 3. Load and preprocess data


df_assoc_naming = preprocess_df(pd.read_csv(ASSOC_NAMING))
df_assoc_usage = preprocess_df(pd.read_csv(ASSOC_USAGE))
df_dec_naming = preprocess_df(pd.read_csv(DEC_NAMING))
df_dec_usage = preprocess_df(pd.read_csv(DEC_USAGE))

# Filter models and drop NaN biases
for df in [df_assoc_naming, df_assoc_usage, df_dec_naming, df_dec_usage]:
    df['model_name'] = df['model_name'].astype(str)
    df = df[df['model_name'].isin(MODELS)].dropna(subset=['bias'])

# Compute statistics
stats_assoc_naming = compute_all_stats(df_assoc_naming)
stats_assoc_usage = compute_all_stats(df_assoc_usage)
stats_dec_naming = compute_all_stats(df_dec_naming)
stats_dec_usage = compute_all_stats(df_dec_usage)

for df_stats in [stats_assoc_naming, stats_assoc_usage, stats_dec_naming, stats_dec_usage]:
    df_stats['setting'] = 'naming' if 'naming' in df_stats['model'].iloc[0] else 'usage'  # quick fix; better to assign manually

stats_assoc_naming['setting'] = 'naming'
stats_assoc_usage['setting'] = 'usage'
stats_dec_naming['setting'] = 'naming'
stats_dec_usage['setting'] = 'usage'


# 4. Print and save tables


def print_full_table(stats_df, title):
    pivot = stats_df.pivot(index='model', columns='dimension', values='mean').reindex(columns=DIM_ORDER)
    print(f"\n{title} (mean bias):")
    print(pivot.round(3))
    detail = stats_df[['model', 'dimension', 'mean', 'ci_low', 'ci_high', 'p_value', 'cohen_d', 'significant']]
    print(f"\n{title} (detailed):")
    print(detail.round(4).to_string(index=False))

print_full_table(stats_assoc_naming, "Association Naming")
print_full_table(stats_assoc_usage, "Association Usage")
print_full_table(stats_dec_naming, "Decision Naming")
print_full_table(stats_dec_usage, "Decision Usage")

# Significance counts
print("\nSignificance counts (p < 0.001):")
for name, df in [("Assoc Naming", stats_assoc_naming), ("Assoc Usage", stats_assoc_usage),
                 ("Dec Naming", stats_dec_naming), ("Dec Usage", stats_dec_usage)]:
    print(f"{name}: {df['significant'].sum()} / {len(df)}")

# Naming vs Usage comparison
def compare_naming_usage(stats_naming, stats_usage, task_name):
    results = []
    for model in MODELS:
        for dim in DIM_ORDER:
            row_naming = stats_naming[(stats_naming['model'] == model) & (stats_naming['dimension'] == dim)]
            row_usage = stats_usage[(stats_usage['model'] == model) & (stats_usage['dimension'] == dim)]
            if len(row_naming) == 0 or len(row_usage) == 0:
                continue
            m1, m2 = row_naming['mean'].values[0], row_usage['mean'].values[0]
            s1, s2 = row_naming['std'].values[0], row_usage['std'].values[0]
            n1, n2 = row_naming['n'].values[0], row_usage['n'].values[0]
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            t = (m1 - m2) / se if se != 0 else 0
            df = n1 + n2 - 2
            p = 2 * (1 - stats.t.cdf(abs(t), df))
            results.append({'model': model, 'dimension': dim, 'task': task_name,
                            'mean_naming': m1, 'mean_usage': m2, 't_stat': t, 'p_value': p,
                            'naming_higher': m1 > m2})
    return pd.DataFrame(results)

comp_assoc = compare_naming_usage(stats_assoc_naming, stats_assoc_usage, 'assoc')
comp_dec = compare_naming_usage(stats_dec_naming, stats_dec_usage, 'dec')

for comp, name in [(comp_assoc, "Association"), (comp_dec, "Decision")]:
    sig = comp[comp['p_value'] < 0.05]
    print(f"{name}: {len(sig)} significant differences; naming higher in {sig['naming_higher'].sum()} cases.")

# Dialect stats for all four tests
if ('language' in df_assoc_naming.columns and
        'language' in df_assoc_usage.columns and
        'language' in df_dec_naming.columns and
        'language' in df_dec_usage.columns):

    def compute_dialect_stats(df, test_name):
        results = []

        for lang in df['language'].unique():
            for dim in DIM_ORDER:
                subset = df[
                    (df['language'] == lang) &
                    (df['dimension'] == dim)
                ].dropna(subset=['bias'])

                if len(subset) < 2:
                    continue

                mean = subset['bias'].mean()
                ci_low, ci_high = bootstrap_ci(subset['bias'])

                results.append({
                    'test': test_name,
                    'language': lang,
                    'dimension': dim,
                    'mean': mean,
                    'ci_low': ci_low,
                    'ci_high': ci_high
                })

        return pd.DataFrame(results)

    dialect_stats_assoc_naming = compute_dialect_stats(
        df_assoc_naming,
        'Association Naming'
    )

    dialect_stats_assoc_usage = compute_dialect_stats(
        df_assoc_usage,
        'Association Usage'
    )

    dialect_stats_dec_naming = compute_dialect_stats(
        df_dec_naming,
        'Decision Naming'
    )

    dialect_stats_dec_usage = compute_dialect_stats(
        df_dec_usage,
        'Decision Usage'
    )

    print("\nDialect-specific biases (Association Naming):")
    print(dialect_stats_assoc_naming.round(4).to_string(index=False))

    print("\nDialect-specific biases (Association Usage):")
    print(dialect_stats_assoc_usage.round(4).to_string(index=False))

    print("\nDialect-specific biases (Decision Naming):")
    print(dialect_stats_dec_naming.round(4).to_string(index=False))

    print("\nDialect-specific biases (Decision Usage):")
    print(dialect_stats_dec_usage.round(4).to_string(index=False))

# Rejection rates
def rejection_rate(df):
    df['rejected'] = df['nones'] >= 1
    return df.groupby('model_name')['rejected'].mean()

df_all = pd.concat([df_assoc_naming, df_assoc_usage, df_dec_naming, df_dec_usage], ignore_index=True)
rates = rejection_rate(df_all)
print("\nRejection rates:")
print(rates.round(4))

# Save CSVs
stats_assoc_naming.to_csv('stats_assoc_naming.csv', index=False)
stats_assoc_usage.to_csv('stats_assoc_usage.csv', index=False)
stats_dec_naming.to_csv('stats_dec_naming.csv', index=False)
stats_dec_usage.to_csv('stats_dec_usage.csv', index=False)
comp_assoc.to_csv('comp_assoc.csv', index=False)
comp_dec.to_csv('comp_dec.csv', index=False)

if ('language' in df_assoc_naming.columns and
        'language' in df_assoc_usage.columns and
        'language' in df_dec_naming.columns and
        'language' in df_dec_usage.columns):

    dialect_stats_assoc_naming.to_csv(
        'dialect_stats_assoc_naming.csv',
        index=False
    )

    dialect_stats_assoc_usage.to_csv(
        'dialect_stats_assoc_usage.csv',
        index=False
    )

    dialect_stats_dec_naming.to_csv(
        'dialect_stats_dec_naming.csv',
        index=False
    )

    dialect_stats_dec_usage.to_csv(
        'dialect_stats_dec_usage.csv',
        index=False
    )

rates.to_csv('rejection_rates.csv')


# 5. Plotting functions (PDF figures)


def plot_main_comparison(stats_naming, stats_usage, title, filename):
    """Main comparison plot: naming vs usage per model (two columns)."""
    n_rows = (len(MODELS) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 4))
    axes = axes.flatten()
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        naming = stats_naming[stats_naming['model'] == model].set_index('dimension')
        usage = stats_usage[stats_usage['model'] == model].set_index('dimension')
        x = np.arange(len(DIM_ORDER))
        width = 0.35
        means_n = [naming.loc[dim, 'mean'] if dim in naming.index else 0 for dim in DIM_ORDER]
        means_u = [usage.loc[dim, 'mean'] if dim in usage.index else 0 for dim in DIM_ORDER]
        err_n = [naming.loc[dim, 'ci_high'] - naming.loc[dim, 'mean'] if dim in naming.index else 0 for dim in DIM_ORDER]
        err_n_low = [naming.loc[dim, 'mean'] - naming.loc[dim, 'ci_low'] if dim in naming.index else 0 for dim in DIM_ORDER]
        err_u = [usage.loc[dim, 'ci_high'] - usage.loc[dim, 'mean'] if dim in usage.index else 0 for dim in DIM_ORDER]
        err_u_low = [usage.loc[dim, 'mean'] - usage.loc[dim, 'ci_low'] if dim in usage.index else 0 for dim in DIM_ORDER]
        ax.bar(x - width/2, means_n, width, yerr=[err_n_low, err_n], capsize=5, label='Naming', color='#414f2f')
        ax.bar(x + width/2, means_u, width, yerr=[err_u_low, err_u], capsize=5, label='Usage', color='#a9bc90')
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(DIM_ORDER, rotation=45, ha='right')
        ax.set_title(model, fontweight='bold')
        ax.set_ylim(-1, 1)
        if idx == 0:
            ax.legend()
    for i in range(len(MODELS), len(axes)):
        axes[i].axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_dialect(dialect_stats, title, ax):
    if dialect_stats.empty:
        ax.axis('off')
        return

    dialects = dialect_stats['language'].unique()
    x = np.arange(len(DIM_ORDER))
    width = 0.8 / len(dialects) if len(dialects) > 0 else 0.1

    for i, lang in enumerate(dialects):
        data = dialect_stats[dialect_stats['language'] == lang]

        means = []
        err_high = []
        err_low = []

        for dim in DIM_ORDER:
            row = data[data['dimension'] == dim]

            if len(row) > 0:
                mean = row['mean'].values[0]
                ci_low = row['ci_low'].values[0]
                ci_high = row['ci_high'].values[0]

                means.append(mean)
                err_low.append(mean - ci_low)
                err_high.append(ci_high - mean)

            else:
                means.append(0)
                err_low.append(0)
                err_high.append(0)

        ax.bar(
            x + i * width - (len(dialects) - 1) * width / 2,
            means,
            width,
            label=lang,
            yerr=[err_low, err_high],
            capsize=3
        )

    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(DIM_ORDER, rotation=45, ha='right')
    ax.set_ylabel('Mean Bias')
    ax.set_title(title)
    ax.legend()

def plot_rejection_rate(rates, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    rates.sort_values().plot(kind='barh', color='steelblue', ax=ax)
    ax.set_xlabel('Rejection Rate')
    ax.set_title('Rejection Rate per Model')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Generate figures
plot_main_comparison(stats_assoc_naming, stats_assoc_usage, 'Association Task', 'assoc_main.pdf')
plot_main_comparison(stats_dec_naming, stats_dec_usage, 'Decision Task', 'dec_main.pdf')

if ('language' in df_assoc_naming.columns and
        'language' in df_assoc_usage.columns and
        'language' in df_dec_naming.columns and
        'language' in df_dec_usage.columns):

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 24)
    )

    plot_dialect(
        dialect_stats_assoc_naming,
        'Association Naming Bias by Dialect',
        axes[0]
    )

    plot_dialect(
        dialect_stats_assoc_usage,
        'Association Usage Bias by Dialect',
        axes[1]
    )

    plot_dialect(
        dialect_stats_dec_naming,
        'Decision Naming Bias by Dialect',
        axes[2]
    )

    plot_dialect(
        dialect_stats_dec_usage,
        'Decision Usage Bias by Dialect',
        axes[3]
    )

    plt.tight_layout()
    plt.savefig(
        'dialect_bias.pdf',
        bbox_inches='tight'
    )
    plt.close()

plot_rejection_rate(rates, 'rejection_rate.pdf')

print("\nAll PDF figures saved.")