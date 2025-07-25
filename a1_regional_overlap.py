import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import (
    load_analysis_dataset,
    COUNTRIES,
    COUNTRY_TO_REGION,
    REGIONS,
    COUNTRIES_WITHIN_REGION,
)
from plot_utils import (
    config_plotting_environment,
    PLOT_COLOR,
    BACKGROUND_COLOR,
    FONT_COLOR,
)

config_plotting_environment()

# Load the dataset
df = load_analysis_dataset()
dev_only = False
if dev_only:
    df = df[df["split"] == "dev"]

# Discard the samples that are not valid in any country
df = df[(df["n_valid"] > 0)]

# Discard the samples that are originally labeled as MSA
df = df[~df["is_msa"]]

# Compute the number of valid regions for each sample
df["n_valid_region"] = df.apply(
    lambda row: len(set([COUNTRY_TO_REGION[c] for c in COUNTRIES if row[c]])),
    axis=1,
)

for region in REGIONS:
    df[region] = df.apply(
        lambda row: any([row[c] for c in COUNTRIES_WITHIN_REGION[region] if c in row]),
        axis=1,
    )

# A check for the correctness of the regional labels
df["n_valid_region_check"] = df.apply(
    lambda row: sum([row[r] for r in REGIONS]), axis=1
)
assert df["n_valid_region_check"].equals(df["n_valid_region"])

assert df["n_valid_region"].max() == 5

n_dialects = [i for i in range(1, 6)]
n_dialects_counts = [sum(df["n_valid_region"] == i) for i in n_dialects]

# height was set to 1!
fig, ax = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6.3 / 2, 0.5)
)
ax.bar(
    n_dialects,
    n_dialects_counts,
    color=[PLOT_COLOR if n_d != 1 else "grey" for n_d in n_dialects],
)
ax.set_facecolor(BACKGROUND_COLOR)

ax.set_xticks(n_dialects)
ax.set_xticklabels(n_dialects, fontsize=7)
for i in n_dialects:
    ax.annotate(
        f"{n_dialects_counts[i - 1]}\n",
        xy=(i - 0.15, n_dialects_counts[i - 1] + max(n_dialects_counts) / 20),
        color=FONT_COLOR,
        fontsize=7,
    )
    ax.annotate(
        f"({100 * n_dialects_counts[i - 1] / len(df):.0f}%)",
        xy=(i - 0.15, n_dialects_counts[i - 1] + max(n_dialects_counts) / 20),
        color=FONT_COLOR,
        fontsize=6,
    )

ax.set_xlabel("No. valid regional dialects", color=FONT_COLOR, fontsize=7)
ax.set_ylabel("No. samples", color=FONT_COLOR, fontsize=7)
ax.spines["bottom"].set_color(FONT_COLOR)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="x", colors=FONT_COLOR)
# ax.tick_params(axis="y", colors=FONT_COLOR)
ax.set_yticks([])
ax.set_yticklabels([])

plt.savefig(
    f"plots/a1_validity_distribution_region{'_dev' if dev_only else ''}.pdf",
    bbox_inches="tight",
)

# Repeat the same analysis for the country-level dialects
fig, ax = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6.3 / 1.5, 0.5)
)
n_dialects = [i for i in range(1, 12)]
n_dialects_counts = [sum(df["n_valid"] == i) for i in n_dialects]

ax.bar(
    n_dialects,
    n_dialects_counts,
    color=[PLOT_COLOR if n_d != 1 else "grey" for n_d in n_dialects],
)
ax.set_facecolor(BACKGROUND_COLOR)

ax.set_xticks(n_dialects)
ax.set_xticklabels(n_dialects, fontsize=7)

for i in n_dialects:
    ax.annotate(
        f"{n_dialects_counts[i - 1]}\n",
        xy=(
            i - (0.4 if n_dialects_counts[i - 1] > 99 else 0.3),
            n_dialects_counts[i - 1] + max(n_dialects_counts) / 20,
        ),
        color=FONT_COLOR,
        fontsize=7,
    )
    ax.annotate(
        f"({100 * n_dialects_counts[i - 1] / len(df):.0f}%)",
        xy=(
            i - (0.4 if n_dialects_counts[i - 1] / len(df) > 0.1 else 0.3),
            n_dialects_counts[i - 1] + max(n_dialects_counts) / 20,
        ),
        color=FONT_COLOR,
        fontsize=6,
    )

ax.set_xlabel("No. valid country-level dialects", color=FONT_COLOR, fontsize=7)
ax.set_ylabel("No. sentences", color=FONT_COLOR, fontsize=7)

ax.spines["bottom"].set_color(FONT_COLOR)
ax.spines["left"].set_visible(False)

ax.tick_params(axis="x", colors=FONT_COLOR)
# ax.tick_params(axis="y", colors=FONT_COLOR)
ax.set_yticks([])
ax.set_yticklabels([])

plt.savefig(
    f"plots/a1_validity_distribution_country{'_dev' if dev_only else ''}.pdf",
    bbox_inches="tight",
)
