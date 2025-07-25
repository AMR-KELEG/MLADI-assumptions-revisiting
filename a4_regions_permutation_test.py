import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

random.seed(42)
from utils import (
    COUNTRIES,
    load_individual_analysis_dataset,
)
from plot_utils import (
    config_plotting_environment,
    PLOT_COLOR,
)

config_plotting_environment(change_plot_frame=False)


def get_shuffled_grouping(group_a, group_b):
    """Form a new shuffle of group of same sizes from the two groups."""
    annotators = group_a + group_b

    # Shuffle the annotators randomly
    random.shuffle(annotators)

    # Split the shuffled annotators into two groups of the same size as the original groups
    shuffled_group_a = annotators[: len(group_a)]
    shuffled_group_b = annotators[len(group_a) :]

    try:
        assert len(shuffled_group_a) == len(
            group_a
        ), "Shuffled group A is not of the same size as the original group A"
        assert len(shuffled_group_b) == len(
            group_b
        ), f"Shuffled group B ({len(shuffled_group_b)}) is not of the same size as the original group B ({len(group_b)})"
    except:
        print("Error in the shuffling")
        print("Group A:", group_a)
        print("Group B:", group_b)
        print("Shuffled Group A:", shuffled_group_a)
        print("Shuffled Group B:", shuffled_group_b)
        raise

    return shuffled_group_a, shuffled_group_b


def compute_ALDi_diff(df, group1, group2):
    # Compute the mean ALDi scores for group1 and group2
    group1_ALDi = df[group1].mean(axis=1)
    group2_ALDi = df[group2].mean(axis=1)

    # Compute the difference between the two groups
    diff = group1_ALDi - group2_ALDi

    # Remove the samples where either group1 or group2 has no ALDi score
    diff = diff[(~group1_ALDi.isnull()) & (~group2_ALDi.isnull())]

    # Check the number of samples with no means!
    n_null = (group1_ALDi.isnull() | group2_ALDi.isnull()).sum()

    return diff.mean(), n_null


def compute_p_value(true_diff, observed_diffs):
    return (
        observed_diffs[observed_diffs <= true_diff].shape[0] / observed_diffs.shape[0]
    )


def run_permutation_test(region_a, region_b):
    # Select the shared dataframe with at least one annotator from each region providing ALDi scores
    shared_df = df[
        (df[f"n_valid_region{region_a}"] >= 1)
        & ((df[f"n_valid_region{region_b}"] >= 1))
    ]

    n_iter = 50000
    mean_diffs = []
    n_nulls = []
    group_a = (
        annotators_1
        if region_a == "1"
        else (
            annotators_2
            if region_a == "2"
            else annotators_3 if region_a == "3" else annotators_4
        )
    )
    group_b = (
        annotators_1
        if region_b == "1"
        else (
            annotators_2
            if region_b == "2"
            else annotators_3 if region_b == "3" else annotators_4
        )
    )

    for it in tqdm(range(n_iter)):
        group1, group2 = get_shuffled_grouping(group_a, group_b)
        diff, n_null = compute_ALDi_diff(shared_df, group1, group2)
        n_nulls.append(n_null)
        mean_diffs.append(diff)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.5))
    axes[0].hist(mean_diffs, bins=50, color=PLOT_COLOR)
    axes[1].hist([n_null / shared_df.shape[0] for n_null in n_nulls])
    axes[1].set_xlim(0, 1)

    true_diff, n_null = compute_ALDi_diff(shared_df, group_a, group_b)
    plt.show(block=False)

    perc_null = 100 * np.array(n_nulls) / shared_df.shape[0]
    return (
        compute_p_value(true_diff, np.array(mean_diffs)),
        true_diff,
        perc_null.mean(),
        perc_null.std(),
        perc_null.min(),
        perc_null.max(),
    )


if __name__ == "__main__":
    df = load_individual_analysis_dataset()
    regions_labels = {
        "1": "Maghreb",
        "2": "Nile Basin",
        "3": "Levant",
        "4": "Gulf+Gulf of Aden",
    }

    # Merge the countries into regional groups
    region_1 = ["Morocco", "Algeria", "Tunisia"]
    region_2 = ["Egypt", "Sudan"]
    region_3 = ["Syria", "Jordan", "Palestine"]
    region_4 = [
        c for c in COUNTRIES if c not in (region_1 + region_2 + region_3)
    ]  # i.e., Gulf and Gulf of Aden

    # Determine the ALDi columns for each region
    annotators_1 = [f"{c}_ann_{ann}_LoD_final" for c in region_1 for ann in "ABC"]
    annotators_2 = [f"{c}_ann_{ann}_LoD_final" for c in region_2 for ann in "ABC"]
    annotators_3 = [f"{c}_ann_{ann}_LoD_final" for c in region_3 for ann in "ABC"]
    annotators_4 = [
        f"{c if c!='Saudi' else 'Saudi_Arabia'}_ann_{ann}_LoD_final"
        for c in region_4
        for ann in "ABC"
    ]

    # Find the number of annotators providing ALDi scores for each region
    df["n_valid_region1"] = df[annotators_1].notnull().sum(axis=1)
    df["n_valid_region2"] = df[annotators_2].notnull().sum(axis=1)
    df["n_valid_region3"] = df[annotators_3].notnull().sum(axis=1)
    df["n_valid_region4"] = df[annotators_4].notnull().sum(axis=1)

    p_values = []
    true_diffs = []
    for group_a_id in "1234":
        p_values.append([])
        true_diffs.append([])
        for group_b_id in "1234":
            if group_a_id == group_b_id:
                p_values[-1].append(1)
                true_diffs[-1].append(0)
            elif group_a_id != group_b_id:
                (
                    p_value,
                    true_diff,
                    perc_null_mean,
                    perc_null_std,
                    perc_null_min,
                    perc_null_max,
                ) = run_permutation_test(group_a_id, group_b_id)
                p_values[-1].append(p_value)
                true_diffs[-1].append(true_diff)

                print(
                    f"'{regions_labels[group_a_id]}' vs '{regions_labels[group_b_id]}':",
                    p_value,
                    true_diff,
                    round(perc_null_min, 2),
                    f"{round(perc_null_mean, 2)}±{round(perc_null_std, 2)}",
                    round(perc_null_max, 2),
                )
            else:
                print("ERROR!")

    # Display the results (differences and p-values)
    print("Differences:")
    print(
        pd.DataFrame(
            true_diffs,
            columns=[regions_labels[b] for b in "1234"],
            index=[regions_labels[a] for a in "1234"],
        )
    )
    print("P-values:")
    print(
        pd.DataFrame(
            p_values,
            columns=[regions_labels[b] for b in "1234"],
            index=[regions_labels[a] for a in "1234"],
        )
    )

    plt.figure(figsize=(6.3 / 2, 6.3 / 2))
    labels = [
        [
            f"{round(diff, 2)} {'*' if p_value < 0.05 else ''}"
            for diff, p_value in zip(diffs, p_values)
        ]
        for diffs, p_values in zip(true_diffs, p_values)
    ]

    sns.heatmap(
        np.array(true_diffs),
        annot=labels,
        fmt="",
        cmap="coolwarm",
        cbar=False,
        xticklabels=[regions_labels[a] for a in "1234"],
        yticklabels=[regions_labels[a] for a in "1234"],
    )
