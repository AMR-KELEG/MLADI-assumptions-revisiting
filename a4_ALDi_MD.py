import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils import COUNTRIES, load_analysis_dataset
from plot_utils import (
    config_plotting_environment,
    PLOT_COLOR,
    FONT_COLOR,
)

import os

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

config_plotting_environment()

full_df = load_analysis_dataset()

dev_only = False
if dev_only:
    full_df = full_df[full_df["split"] == "dev"]

full_df = full_df[~full_df["is_msa"]]

MD_scores, STD_scores, SE_scores, n_commonly_valid_values = ([], [], [], [])
for country_row in COUNTRIES:
    (
        country_MD_scores,
        country_STD_scores,
        country_SE_scores,
        country_n_commonly_valid_values,
    ) = ([], [], [], [])
    for country_col in COUNTRIES:
        if country_row == country_col:
            country_n_commonly_valid_values.append(full_df[country_row].sum())
            country_MD_scores.append(0)
            country_STD_scores.append(0)
            country_SE_scores.append(0)
            continue
        valid_ab_df = full_df[(full_df[country_row]) & (full_df[country_col])]
        diff = valid_ab_df[f"{country_row}_ALDi"] - valid_ab_df[f"{country_col}_ALDi"]
        MD_score = diff.mean()
        STD_score = diff.std()
        SE_score = diff.std() / np.sqrt(diff.shape[0])
        country_n_commonly_valid_values.append(valid_ab_df.shape[0])
        country_MD_scores.append(MD_score)
        country_STD_scores.append(STD_score)
        country_SE_scores.append(SE_score)
    MD_scores.append(country_MD_scores)
    STD_scores.append(country_STD_scores)
    SE_scores.append(country_SE_scores)
    n_commonly_valid_values.append(country_n_commonly_valid_values)

MD_labels = [
    [
        f"{round(MD_value, 2)}\n{n_commonly_valid_value}"
        for MD_value, SE_value, n_commonly_valid_value in zip(
            MD_values, SE_values, n_commonly_valid_values_list
        )
    ]
    for MD_values, SE_values, n_commonly_valid_values_list in zip(
        MD_scores, SE_scores, n_commonly_valid_values
    )
]

STD_labels = [
    [f"±{round(STD_value, 2)}" for STD_value in STD_values] for STD_values in STD_scores
]

SE_labels = [
    [f"±{round(SE_value, 2)}" for SE_value in SE_values] for SE_values in SE_scores
]

country_samples = [sum(full_df[c]) for c in COUNTRIES]

percentage_commonly_valid_values = [
    [
        round(100 * value / country_samples[row], 1)
        for col, value in enumerate(n_commonly_valid_values_list)
    ]
    for row, n_commonly_valid_values_list in enumerate(n_commonly_valid_values)
]
percentage_commonly_valid_labels = [
    [
        f"{round(100*value/country_samples[row], 1)}%\n({value})"
        for col, value in enumerate(n_commonly_valid_values_list)
    ]
    for row, n_commonly_valid_values_list in enumerate(n_commonly_valid_values)
]

mask = np.eye(len(COUNTRIES), dtype=bool)

for metric, scores, labels, cmap in [
    ("MD", MD_scores, MD_labels, "PuOr"),
]:
    figure_width = 6.3 / 2 + 0.1
    figure_height = 6.3 / 3
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(figure_width, figure_height),
        gridspec_kw={
            "width_ratios": [
                0.15,
                (figure_height - 0.15) / 2,
            ],
            "height_ratios": [1],
            "wspace": 0.2,
            "hspace": 0.1,
        },
        tight_layout=True,
        sharey=True,
    )

    ax = sns.heatmap(
        np.array(scores),
        cmap=cmap,
        mask=mask if metric != "N" else None,
        cbar=True,
        annot=labels,
        xticklabels=COUNTRIES,
        yticklabels=COUNTRIES,
        fmt="",
        annot_kws={"size": 4},
        ax=axes[1],
    )

    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=4)

    for i in range(len(scores) + 1):
        ax.axhline(i, color="white", lw=1)

    axes[1].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    axes[1].set_yticklabels(COUNTRIES, fontsize=5)
    axes[1].set_xticklabels(COUNTRIES, fontsize=5, rotation=45)
    axes[1].set_title(
        f"$MD(r,c)=\\frac{{1}}{{N_{{rc}}}}\\sum_{{i=1}}^{{i=N_{{rc}}}}{{ALDi_{{r}}[i] - ALDi_{{c}}[i]}}$",
        fontdict={"size": 6},
    )

    axes[0].barh(
        COUNTRIES,
        country_samples,
        color=PLOT_COLOR,
        align="edge",
    )
    for i, r in enumerate(COUNTRIES):
        axes[0].annotate(
            f"{country_samples[i]}",
            xy=(country_samples[i] + max(country_samples) / 20, i + 0.6),
            color=FONT_COLOR,
            fontsize=4.5,
            rotation=0,
        )

    axes[0].set_xlabel("No. valid samples", color=FONT_COLOR, fontsize=5)
    axes[0].set_yticklabels(COUNTRIES, fontsize=5, rotation=0)
    axes[0].set_xticklabels([int(v) for v in axes[0].get_xticks()], fontsize=6)
    axes[0].spines["bottom"].set_visible(False)
    axes[0].set_xticks([])
    axes[0].set_xticklabels([])

    plt.tight_layout()
    fig.tight_layout()

    figure_name = (
        f"plots/a4_ALDi_perceptions_{metric}{'_dev' if dev_only else ''}_h.pdf"
    )
    plt.savefig(figure_name, bbox_inches="tight")
