import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from utils import load_analysis_dataset
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

df["n_tokens"] = df["sentence"].apply(lambda s: len(s.split()))
df["ALDi_agg"] = df["ALDi_detailed"].apply(lambda l: np.mean(l))

print(df[["n_tokens", "n_valid"]].corr(method="spearman"))

sentence_lengths_bins = [(1, 5), (6, 15), (16, 25), (26, 35), (36, 100)]
n_bins = len(sentence_lengths_bins)

figure, axes = plt.subplots(
    nrows=1, ncols=n_bins, figsize=(6.12, 1), sharex=False, sharey=True
)

for i, (ax, (low_th, high_th)) in enumerate(zip(axes, sentence_lengths_bins)):
    bin_df = df[((df["n_tokens"] >= low_th) & (df["n_tokens"] <= high_th))]
    freq, bins, patches = ax.hist(
        bin_df["n_valid"],
        bins=11,
        color=PLOT_COLOR,
        log=True,
        range=(df["n_valid"].min(), df["n_valid"].max() + 1),
        alpha=1,
        edgecolor="black",
        linewidth=0.5,
        align="mid",
    )

    bin_centers = np.diff(bins) * 0.5 + bins[:-1]

    # Annotate the bars
    for fr, x, patch in zip(freq, bin_centers, patches):
        height = int(fr)
        ax.annotate(
            "{}".format(height),
            xy=(x, height),  # top left corner of the histogram bar
            xytext=(0, 0.2),  # offsetting label position above its bar
            textcoords="offset points",  # Offset (in points) from the *xy* value
            ha="center",
            va="bottom",
            fontsize=3,
        )

    ax.set_title(f"No. tokens in [{low_th}, {high_th}]", fontsize=7)
    ax.set_xlabel(f"No. valid dialects", fontsize=6)

    ax.set_xticks(
        ticks=[i + 0.5 for i in range(1, 12)],
        labels=[i if i % 2 == 1 else "" for i in range(1, 12)],
        rotation=0,
        fontsize=6,
    )
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_yticklabels([1, 10, 100], fontsize=6)
    # ax.set_ylim(0, 250)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if i == 0:
        ax.set_ylabel("No. samples", fontsize=6, color=FONT_COLOR)

figure.tight_layout()

figure.savefig(
    f"plots/a2_validity_length{'_dev' if dev_only else ''}.pdf", bbox_inches="tight"
)

# Repeat the analysis for ALDi scores
print("Manual ALDi scores:")
print(df[["ALDi_agg", "n_valid"]].corr(method="spearman"))
print("Automatic ALDi scores:")
print(df[["automatic_sentence_ALDi", "n_valid"]].corr(method="spearman"))

ALDi_bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
n_bins = len(ALDi_bins)

figure, axes = plt.subplots(
    nrows=1, ncols=n_bins, figsize=(6.12, 1), sharex=False, sharey=True
)

n_samples = 0
for i, (ax, (low_th, high_th)) in enumerate(zip(axes, ALDi_bins)):
    bin_df = df[((df["ALDi_agg"] >= low_th) & (df["ALDi_agg"] < high_th))]
    n_samples += bin_df.shape[0]
    freq, bins, patches = ax.hist(
        bin_df["n_valid"],
        bins=11,
        color=PLOT_COLOR,
        log=True,
        range=(df["n_valid"].min(), df["n_valid"].max() + 1),
        alpha=1,
        edgecolor="black",
        linewidth=0.5,
        align="mid",
    )
    bin_centers = np.diff(bins) * 0.5 + bins[:-1]

    # Annotate the bars
    for fr, x, patch in zip(freq, bin_centers, patches):
        height = int(fr)
        ax.annotate(
            "{}".format(height),
            xy=(x, height),  # top left corner of the histogram bar
            xytext=(0, 0.2),  # offsetting label position above its bar
            textcoords="offset points",  # Offset (in points) from the *xy* value
            ha="center",
            va="bottom",
            fontsize=3,
        )

    ax.set_xticks(
        ticks=[i + 0.5 for i in range(1, 12)],
        labels=[i if i % 2 == 1 else "" for i in range(1, 12)],
        rotation=0,
        fontsize=6,
    )
    ax.set_facecolor(BACKGROUND_COLOR)

    ax.set_yticklabels([1, 10, 100], fontsize=6)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if i == 0:
        ax.set_ylabel("No. samples", fontsize=6)
    ax.set_title(
        f"ALDi in [{low_th}, {min(high_th, 1)}{'[' if low_th < 0.7 else ']'}",
        fontsize=7,
    )
    ax.set_xlabel(f"No. valid dialects", fontsize=6)

print("No of samples:", n_samples)
figure.tight_layout()

figure.savefig(
    f"plots/a2_validity_ALDi{'_dev' if dev_only else ''}.pdf", bbox_inches="tight"
)
