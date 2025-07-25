import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from utils import (
    REGIONS,
    COUNTRIES_WITHIN_REGION,
    load_analysis_dataset,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

from plot_utils import (
    config_plotting_environment,
    PLOT_COLOR,
)

config_plotting_environment()

df = load_analysis_dataset()
df = df[~df["is_msa"]]
df["average_ALDi"] = df["ALDi_detailed"].apply(lambda l: sum(l) / len(l) if l else None)

for region in REGIONS:
    df[region] = df.apply(
        lambda row: any([row[c] for c in COUNTRIES_WITHIN_REGION[region] if c in row]),
        axis=1,
    )

df["n_valid_region"] = df[REGIONS].sum(axis=1)
df["regions"] = df.apply(lambda row: [r for r in REGIONS if row[r]], axis=1)

n_samples_i = {}
for i in range(1, 6):
    n_samples_i[i] = []
    for region in REGIONS:
        current_df = df[(df["n_valid_region"] == i) & (df[region])].copy(deep=True)
        current_df["other_regions"] = current_df["regions"].apply(
            lambda l: ", ".join([r for r in l if r != region])
        )
        common_groups = Counter(current_df["other_regions"]).most_common()
        common_groups = ", ".join([f"{g[0]} ({g[1]})" for g in common_groups])
        print(f"{i}) {region} ({len(current_df)}):\n{common_groups}\n")
        n_samples_i[i].append(len(current_df))
    print()

region_samples = [df[df[r]].shape[0] for r in REGIONS]

fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(6.3 / 2, 3), sharey=True, sharex=False, tight_layout=True
)
skip_row = 2
skip_col = 1
axes[skip_row][skip_col].axis("off")
axes = [ax for row in axes for ax in row if ax != axes[skip_row][skip_col]]
for i, ax in zip(range(len(REGIONS)), axes):
    region_values = [n_samples_i[j][i] for j in range(1, 6)]
    ax.bar(range(1, 6), region_values, color=PLOT_COLOR)
    # annotate the bars with the number of samples per bar
    for j, n_samples in enumerate(region_values):
        ax.text(
            j + 1, n_samples + 5, f"{n_samples}", ha="center", va="bottom", fontsize=7
        )
        # ax.text(j+1, n_samples+25, f"{n_samples}", ha="center", va="bottom", fontsize=7)
        # ax.text(j+1, n_samples, f"{round(100 * n_samples/region_samples[i])}%", ha="center", va="bottom", fontsize=4)
    ax.set_title(f"{REGIONS[i]}'s {region_samples[i]} samples\n", fontsize=7)
    if i == 0 or i == 2 or i == 4:
        ax.set_ylabel("No. samples", fontsize=7)
    ax.set_xticks(range(1, 6), [str(i) for i in range(1, 6)], fontsize=7)
    ax.set_yticklabels([int(v) for v in ax.get_yticks()], fontsize=7)
    ax.set_xlabel("No. valid regions", fontsize=7)


fig.savefig(
    "plots/a1_regions_dialects_distribution_larger_font.pdf", bbox_inches="tight"
)
