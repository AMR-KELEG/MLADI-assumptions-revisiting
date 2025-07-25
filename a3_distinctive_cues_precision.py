import re
import glob
import pandas as pd
from tqdm import tqdm
from utils import (
    load_analysis_dataset,
    COUNTRIES,
    normalize_text,
    load_distinctive_words,
)
from collections import Counter

tqdm.pandas()


def identify_country_cues_in_tokens(tokens, country, distinctive_cues_lists):
    cues = distinctive_cues_lists[country]
    return [c for c in cues if c in tokens]


def identify_cues_in_tokens(tokens, cues_list):
    ngram_cues = {}
    for cue in cues_list:
        ngram = len(cue.split())
        if ngram not in ngram_cues:
            ngram_cues[ngram] = []
        ngram_cues[ngram].append(cue)

    cues_in_tokens = []
    for ngram in sorted(ngram_cues.keys()):
        cues = ngram_cues[ngram]
        for i in range(len(tokens) - ngram + 1):
            if " ".join(tokens[i : i + ngram]) in cues:
                cues_in_tokens.append(" ".join(tokens[i : i + ngram]))
    return cues_in_tokens


def compute_region_stats(df, dataset, region, region_cues_list, apply_geolabel=False):
    # Generate the stats object
    region_column_name = "Country" if dataset in ["DarijaBERT", "TWT15DA"] else "Region"
    region_stats = {region_column_name: region}

    region_name = (
        "Egypt" if region == "EGY" else ("Iraq" if region == "IRQ" else region)
    )
    n_valid_column_name = (
        "n_valid" if dataset in ["DarijaBERT", "TWT15DA"] else "n_valid_region"
    )
    columns = ["sentence", "is_msa", n_valid_column_name, "automatic_sentence_ALDi"] + [
        c for c in df.columns if region_name in c or region in c
    ]

    regions_valid_samples = df[region_name].sum()
    geolabel_column_name = (
        "geolabel" if dataset in ["DarijaBERT", "TWT15DA"] else "geolabel_region"
    )
    df = (
        df
        if not apply_geolabel
        else (
            df[df[geolabel_column_name] == "Saudi_Arabia"]
            if region == "Saudi"
            else df[df[geolabel_column_name] == region]
        )
    )

    region_cues_column = (
        f"{region}_cues"
        if dataset == "TWT15DA"
        else (
            f"{region}_DART_cues"
            if dataset == "DART"
            else (
                f"{region}_DarijaBERT_cues"
                if dataset == "DarijaBERT"
                else f"{region}_DIAL2MSA_cues"
            )
        )
    )
    region_cue_df = df.loc[df[region_cues_column].apply(lambda l: len(l) > 0), columns]

    # Only perform the analysis to the DA samples!
    total_valid_sentences_in_region = df[df[region_name]].shape[0]
    n_samples_with_cues = region_cue_df.shape[0]
    n_valid_sentences_with_cues_in_region = region_cue_df[region_name].sum()
    n_valid_sentences_with_cues_in_region_single_label = region_cue_df[
        (region_cue_df[region_name]) & (region_cue_df[n_valid_column_name] == 1)
    ].shape[0]

    if not apply_geolabel:
        assert total_valid_sentences_in_region == regions_valid_samples

    matching_cues = set(
        [
            cue
            for cues_list in region_cue_df[region_cues_column].tolist()
            for cue in cues_list
        ]
    )

    region_stats[region_column_name] = f"{region_stats[region_column_name]}"

    region_stats["Matching"] = f"{n_samples_with_cues}"

    region_stats["Matching and Valid"] = n_valid_sentences_with_cues_in_region

    region_stats["Matching and Exc"] = (
        n_valid_sentences_with_cues_in_region_single_label
    )

    region_stats["DA Valid Samples"] = regions_valid_samples
    region_stats["Precision"] = (
        round(n_valid_sentences_with_cues_in_region / n_samples_with_cues, 2)
        if n_samples_with_cues > 0
        else "-"
    )
    region_stats["Disctiveness"] = (
        round(
            n_valid_sentences_with_cues_in_region_single_label / n_samples_with_cues, 2
        )
        if n_samples_with_cues > 0
        else "-"
    )

    region_stats["Recall"] = round(
        n_valid_sentences_with_cues_in_region / regions_valid_samples,
        2,
    )

    region_stats["Cues"] = len(region_cues_list)
    region_stats["Matching Cues"] = len(matching_cues)
    return region_stats


def fix_latex(table):
    table = re.sub(
        r"textbackslash ",
        "",
        table,
    )
    table = re.sub(
        r"\\[{]",
        "{",
        table,
    )
    table = re.sub(
        r"\\[}]",
        "}",
        table,
    )

    table = re.sub(r"\s+0[.]", " .", table)
    return table


def main():
    # Load the data
    df = load_analysis_dataset()

    # Only keep the DA samples
    df = df[df["is_msa"] == False]

    df["normalized_sentence"] = df["sentence"].apply(lambda s: normalize_text(s))
    # Tokenize the text
    df["tokens"] = df["normalized_sentence"].progress_apply(lambda s: s.split())

    # TWT15DA cues
    print("TWT15DA!")
    countries_map = {"Uae": "UAE", "Tunis": "Tunisia"}

    distinctive_cues_lists = {
        filename.split("list_")[1]
        .split("_pmi")[-2]
        .capitalize(): load_distinctive_words(
            "TWT15DA", filename.split("list_")[1].split("_pmi")[-2]
        )
        for filename in sorted(glob.glob("data/Twt15DA_Lists/list*.txt"))
    }
    countries_with_distinctive_cues = list(distinctive_cues_lists.keys())

    # Make sure the cues of TWT15DA are single tokens only!
    for country in countries_with_distinctive_cues:
        assert not any([" " in cue for cue in distinctive_cues_lists[country]]), country

    for country in countries_with_distinctive_cues:
        df[f"{countries_map.get(country, country)}_cues"] = df["tokens"].apply(
            lambda tokens: identify_country_cues_in_tokens(
                tokens, country, distinctive_cues_lists
            )
        )

    for apply_geolabel in [False, True]:
        # Compute the precision/recall of samples with cues
        country_precision_recall_scores = []
        # TODO: Programmatically find the common countries between the two lists
        for country in COUNTRIES:
            # some countries are not part of the TWTDA15 lists
            if country in ["Sudan", "Palestine"]:
                continue

            country_stats = compute_region_stats(
                df,
                "TWT15DA",
                country,
                (
                    distinctive_cues_lists[country]
                    if country != "Tunisia"
                    else distinctive_cues_lists["Tunis"]
                ),
                apply_geolabel=apply_geolabel,
            )
            country_precision_recall_scores.append(country_stats)

        print(
            fix_latex(
                pd.DataFrame(country_precision_recall_scores).to_latex(
                    index=False,
                ),
            )
        )

    # DART cues

    print("DART!")
    REGIONS = ["EGY", "IRQ", "MGH", "LEV", "GLF"]

    # Compute the regional level validity labels for the three regions
    df["MGH"] = df["Algeria"] | df["Tunisia"] | df["Morocco"]
    df["LEV"] = df["Palestine"] | df["Syria"] | df["Jordan"]
    df["GLF"] = df["Saudi"]
    df["OTHERS"] = df["Yemen"] | df["Sudan"]

    df["n_valid_region"] = (
        df["MGH"].astype(int)
        + df["LEV"].astype(int)
        + df["GLF"].astype(int)
        + df["Egypt"].astype(int)
        + df["Iraq"].astype(int)
        + df["OTHERS"].astype(int)
    )

    DART_country_to_region_map = {
        "Egypt": "EGY",
        "Sudan": "OTHERS",
        "Iraq": "IRQ",
        "Algeria": "MGH",
        "Tunisia": "MGH",
        "Morocco": "MGH",
        "Libya": "OTHERS",
        "Syria": "LEV",
        "Jordan": "LEV",
        "Lebanon": "LEV",
        "Palestine": "LEV",
        "Yemen": "OTHERS",
        "Saudi_Arabia": "GLF",
        "UAE": "GLF",
    }

    df["geolabel_region"] = df["geolabel"].apply(
        lambda label: DART_country_to_region_map[label]
    )

    # Load the cues
    region_DART_cues = {
        region: load_distinctive_words("DART", region) for region in REGIONS
    }

    for region in REGIONS:
        df[f"{region}_DART_cues"] = df["tokens"].apply(
            lambda tokens: identify_cues_in_tokens(tokens, region_DART_cues[region])
        )

    for region_list in [
        ["EGY", "IRQ"],
        [r for r in REGIONS if r not in ["EGY", "IRQ"]],
    ]:
        for apply_geolabel in [False, True]:
            region_precision_recall_scores = []
            for region in region_list:
                region_stats = compute_region_stats(
                    df,
                    "DART",
                    region,
                    region_DART_cues[region],
                    apply_geolabel=apply_geolabel,
                )
                region_precision_recall_scores.append(region_stats)

            print(
                fix_latex(
                    pd.DataFrame(region_precision_recall_scores).to_latex(
                        index=False,
                    )
                )
            )

    # DIAL2MSA
    print("DIAL2MSA!")
    DIAL2MSA_REGIONS = ["Egyptian", "Levantine", "Gulf", "Maghrebi"]
    REGIONS_ABBRV_map = {
        "Egyptian": "EGY",
        "Levantine": "LEV",
        "Gulf": "GLF",
        "Maghrebi": "MGH",
    }

    # It is not clear if Iraqi is part of Gulf or not!
    region_DIAL2MSA_cues = {
        REGIONS_ABBRV_map[region]: load_distinctive_words("DIAL2MSA", region)
        for region in DIAL2MSA_REGIONS
    }

    for region in DIAL2MSA_REGIONS:
        mapped_region_name = REGIONS_ABBRV_map[region]
        df[f"{mapped_region_name}_DIAL2MSA_cues"] = df["tokens"].apply(
            lambda tokens: identify_cues_in_tokens(
                tokens, region_DIAL2MSA_cues[mapped_region_name]
            )
        )

    for region_list in [
        ["Egyptian"],
        [r for r in DIAL2MSA_REGIONS if r not in ["Egyptian"]],
    ]:
        for apply_geolabel in [False, True]:
            region_precision_recall_scores = []
            for region in region_list:
                mapped_region_name = REGIONS_ABBRV_map[region]
                region_stats = compute_region_stats(
                    df,
                    "DIAL2MSA",
                    mapped_region_name,
                    region_DIAL2MSA_cues[mapped_region_name],
                    apply_geolabel=apply_geolabel,
                )
                region_precision_recall_scores.append(region_stats)

            print(
                fix_latex(
                    pd.DataFrame(region_precision_recall_scores).to_latex(
                        index=False,
                    ),
                )
            )

    # Export the data with the cues!
    df.to_csv("data/df_with_cues.csv", index=False)

    countries_matching_cues = {"country": [], "cues": []}
    for country in COUNTRIES:
        if country in ["Sudan", "Palestine"]:
            continue
        matching_cues = Counter(sum(df[f"{country}_cues"].tolist(), []))
        countries_matching_cues["country"].append(country)
        countries_matching_cues["cues"].append(matching_cues.most_common())
    pd.DataFrame(countries_matching_cues).to_csv(
        "data/TWT15DA_countries_matching_cues.csv"
    )

    regions_matching_cues = {"region": [], "cues": []}
    for region in REGIONS:
        matching_cues = Counter(sum(df[f"{region}_DART_cues"].tolist(), []))
        regions_matching_cues["region"].append(region)
        regions_matching_cues["cues"].append(matching_cues.most_common())

    pd.DataFrame(regions_matching_cues).to_csv("data/DART_regions_matching_cues.csv")


if __name__ == "__main__":
    main()
