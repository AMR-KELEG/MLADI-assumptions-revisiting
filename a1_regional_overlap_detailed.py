import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import (
    load_analysis_dataset,
    COUNTRIES,
    COUNTRY_TO_REGION,
    REGIONS,
    REGIONS_ABB,
    COUNTRIES_WITHIN_REGION,
)
from plot_utils import (
    config_plotting_environment,
)

config_plotting_environment()

# Load the dataset
df = load_analysis_dataset()
dev_only = False
add_more_spacing = True
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

n_region_samples = {2: 173, 3: 141, 4: 114}

# figure_height = 2.5
figure_height = 3
figure_width = 6.3 / 2
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(figure_width, figure_height),
    gridspec_kw={
        "width_ratios": [1],
        "height_ratios": [0.45, figure_height - 0.45],
        "wspace": 0.5,
        "hspace": 0.35,
    },
    sharey=False,
)

correction_term = 0

four_region_samples = [
    df[(~df[r]) & (df["n_valid_region"] == 4)].shape[0] for r in REGIONS[::-1]
]
labels = [
    [REGIONS_ABB[r] for r in REGIONS[::-1] if r != r_ex] for r_ex in REGIONS[::-1]
]
labels = [
    f"{count}\n{r_l[0]} {r_l[1]}\n{r_l[2]} {r_l[3]}"
    for count, r_l in zip(four_region_samples, labels)
]

sns.heatmap(
    np.array(
        [[0] + four_region_samples] if add_more_spacing else [four_region_samples]
    ),
    cmap="Purples",
    cbar=False,
    annot=[[""] + labels] if add_more_spacing else [labels],
    vmin=0,
    vmax=62,
    square=True,
    fmt="",
    ax=axes[0],
    annot_kws={"fontsize": 4.5},
)
axes[0].set_yticklabels([])
axes[0].set_xticklabels(
    ([""] if add_more_spacing else []) + [f"¬{REGIONS_ABB[r]}" for r in REGIONS[::-1]],
    fontsize=4.5,
)
axes[0].set_xlabel(f"4-region samples ({n_region_samples[4]} in total)", fontsize=5)
axes[0].xaxis.set_label_position("top")
axes[0].xaxis.tick_top()
axes[0].tick_params(top=False, bottom=False, left=False, right=False, pad=1)

for n_regions in [2, 3]:
    # Generate the overlap heatmap
    regional_overlap_values = []
    if n_regions == 2:
        for row in REGIONS:
            region_overlap = []
            for col in REGIONS:
                if row == col:
                    region_overlap.append(0)
                    continue
                n_common_samples = df[
                    (df[row]) & (df[col]) & (df["n_valid_region"] == n_regions)
                ].shape[0]
                region_overlap.append(n_common_samples - correction_term)
            regional_overlap_values.append(region_overlap)

    elif n_regions == 3:
        for row in REGIONS[::-1]:
            region_overlap = []
            for col in REGIONS[::-1]:
                if row == col:
                    region_overlap.append(0)
                    continue
                n_common_samples = df[
                    (~df[row]) & (~df[col]) & (df["n_valid_region"] == n_regions)
                ].shape[0]
                region_overlap.append(n_common_samples - correction_term)
            regional_overlap_values.append(region_overlap)

    else:
        exit("ERROR!")

    if n_regions == 2:
        overlap_labels = [
            [
                f"{value}\n{REGIONS_ABB[REGIONS[row_index]]} {REGIONS_ABB[REGIONS[col_index]]}"
                for col_index, value in enumerate(overlap_values)
            ]
            for row_index, overlap_values in enumerate(regional_overlap_values)
        ]
        if add_more_spacing:
            regional_overlap_values = [[0 for _ in range(6)]] + [
                l + [0] for l in regional_overlap_values
            ]
            overlap_labels = [["" for _ in range(6)]] + [
                l + [""] for l in overlap_labels
            ]

    elif n_regions == 3:
        overlap_labels = [
            [
                f"{value}\n{' '.join([REGIONS_ABB[r] for r in REGIONS[::-1] if r not in [REGIONS[::-1][col_index], REGIONS[::-1][row_index]]])}"
                for col_index, value in enumerate(overlap_values)
            ]
            for row_index, overlap_values in enumerate(regional_overlap_values)
        ]
        overlap_labels = [
            [re.sub(r"^([^ ]+ [^ ]+) (.*$)", r"\1\n\2", l) for l in ovelap_list]
            for ovelap_list in overlap_labels
        ]

        if add_more_spacing:
            regional_overlap_values = [[0] + l for l in regional_overlap_values] + [
                [0 for _ in range(6)]
            ]
            overlap_labels = [[""] + l for l in overlap_labels] + [
                ["" for _ in range(6)]
            ]

    if n_regions == 2:
        region_labels = [REGIONS_ABB[r] for r in REGIONS]
    else:
        region_labels = [f"¬{REGIONS_ABB[r]}" for r in REGIONS[::-1]]

    if n_regions == 2:
        mask = [
            [
                row - col < (2 if add_more_spacing else 1)
                for col in range(len(REGIONS) + int(add_more_spacing))
            ]
            for row in range(len(REGIONS) + int(add_more_spacing))
        ]
    else:
        mask = [
            [
                col - row < (2 if add_more_spacing else 1)
                for col in range(len(REGIONS) + int(add_more_spacing))
            ]
            for row in range(len(REGIONS) + int(add_more_spacing))
        ]

    mask = np.array(mask)
    print(
        mask.shape,
        np.array(regional_overlap_values).shape,
        np.array(overlap_labels).shape,
    )
    ax = sns.heatmap(
        np.array(regional_overlap_values),
        cmap="Purples",
        mask=mask,
        cbar=False,
        annot=overlap_labels,
        xticklabels=region_labels + ([""] if add_more_spacing else []),
        yticklabels=region_labels + ([""] if add_more_spacing else []),
        vmin=0,
        vmax=62,
        square=True,
        fmt="",
        ax=axes[1],
        annot_kws={"fontsize": 4.5},
    )

    ax_top = axes[1].secondary_xaxis("top" if n_regions == 3 else "bottom")
    ax_top.set_xticks(axes[1].get_xticks())
    ax_top.set_xticklabels(
        (
            [""] * (1 + int(add_more_spacing)) + region_labels[1:]
            if n_regions == 3
            else region_labels[:-1] + [""] * (1 + int(add_more_spacing))
        ),
        fontsize=4.5,
    )
    ax_top.set_xlabel(
        f"{n_regions}-region samples ({n_region_samples[n_regions]} in total)",
        fontsize=5,
        loc="left" if n_regions == 2 else "right",
    )

    ax_right = axes[1].secondary_yaxis("right" if n_regions == 3 else "left")
    ax_right.set_yticks(axes[1].get_yticks())
    ax_right.set_yticklabels(
        (
            [""] * (1 + int(add_more_spacing)) + region_labels[1:]
            if n_regions == 2
            else region_labels[:-1] + [""] * (1 + int(add_more_spacing))
        ),
        fontsize=4.5,
    )

    for a in [ax_right, ax_top]:
        for spine in a.spines.values():
            spine.set_visible(False)
        a.tick_params(top=False, bottom=False, left=False, right=False, pad=1)

    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[1].tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
    )


plt.savefig(
    f"plots/a1_validity_regional_overlap_detailed{'_dev' if dev_only else ''}_h.pdf",
    bbox_inches="tight",
)
