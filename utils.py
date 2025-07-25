import re
import numpy as np
import pandas as pd
import unicodedata
from glob import glob

COUNTRY_TO_REGION = {
    "Algeria": "Maghreb",
    "Egypt": "Nile Basin",
    "Iraq": "Gulf",
    "Jordan": "Levant",
    "Morocco": "Maghreb",
    "Palestine": "Levant",
    "Sudan": "Nile Basin",
    "Syria": "Levant",
    "Tunisia": "Maghreb",
    "Yemen": "Gulf of Aden",
    "Saudi": "Gulf",
}

COUNTRIES = [
    "Morocco",
    "Algeria",
    "Tunisia",
    "Egypt",
    "Sudan",
    "Jordan",
    "Palestine",
    "Syria",
    "Iraq",
    "Yemen",
    "Saudi",
]

REGIONS = ["Maghreb", "Nile Basin", "Levant", "Gulf", "Gulf of Aden"]
REGIONS_ABB = {
    "Maghreb": "MG",
    "Nile Basin": "NL",
    "Levant": "LV",
    "Gulf": "GL",
    "Gulf of Aden": "AD",
}
COUNTRIES_WITHIN_REGION = {
    region: [c for c in COUNTRIES if COUNTRY_TO_REGION[c] == region]
    for region in REGIONS
}


def load_country_df(filename):
    """Load the 'aggregated' labels for a single country."""

    country = filename.split("/")[-1][:-4]
    ALDi_column = f"{country}_ALDi"
    single_ALDi_column = f"{country}_ALDi_list"
    automatic_sentence_ALDi = "automatic_sentence_ALDi"
    geolocated_label = "geolabel"

    df = pd.read_csv(filename, sep="\t")
    df[country] = df["n_valid_final"].apply(lambda n: n >= 2)

    df[single_ALDi_column] = df[f"{country}_LoD_aggregated"].apply(
        lambda l: [float(a) for a in l[1:-1].split(",")] if l[1:-1] else None
    )

    # Only consider the ALDi scores if the sentence is valid in this country!
    df[single_ALDi_column] = df.apply(
        lambda row: row[single_ALDi_column] if row[country] else None, axis=1
    )

    # Compute the mean ALDi score for each country
    df[ALDi_column] = df[single_ALDi_column].apply(lambda l: np.mean(l) if l else None)

    df[automatic_sentence_ALDi] = df["ALDi"]
    df[geolocated_label] = df["label"]

    return df[
        [
            "sentence",
            "is_msa",
            geolocated_label,
            automatic_sentence_ALDi,
            country,
            ALDi_column,
            single_ALDi_column,
        ]
    ]


def load_individual_annotations_df(filename):
    """Load the individual annotations for a single country with the majority-vote validity label."""
    country = filename.split("/")[-1][:-4]

    df = pd.read_csv(filename, sep="\t")
    df[country] = df["n_valid_final"].apply(lambda n: n >= 2)

    df = df[
        ["sentence", "is_msa"]
        + [c for c in df.columns if c.endswith("_final") or c.endswith("final_comment")]
    ].copy(deep=True)
    return df


def load_individual_analysis_dataset():
    """Load the individual annotations for all countries with the majority-vote validity labels."""
    country_dfs = [
        load_individual_annotations_df(filename)
        for filename in glob("data/analysis/main_task/*.tsv")
    ]

    full_df = pd.concat(
        country_dfs[0:1]
        + [
            df[[c for c in df.columns if c not in ["sentence", "is_msa"]]]
            for df in country_dfs[1:]
        ],
        axis=1,
    )

    return full_df


def load_analysis_dataset(full_version=True):
    """Load the 'aggregated' labels for all countries."""
    automatic_sentence_ALDi = "automatic_sentence_ALDi"
    geolocated_label = "geolabel"
    country_dfs = [
        load_country_df(filename) for filename in glob("data/analysis/main_task/*.tsv")
    ]

    full_df = pd.concat(
        country_dfs[0:1]
        + [
            df[
                [
                    c
                    for c in df.columns
                    if c
                    not in [
                        "sentence",
                        "is_msa",
                        geolocated_label,
                        automatic_sentence_ALDi,
                    ]
                ]
            ]
            for df in country_dfs[1:]
        ],
        axis=1,
    )

    if full_version:
        full_df["n_valid"] = full_df.apply(
            lambda row: sum([row[c] == True for c in row.keys() if c in COUNTRIES]),
            axis=1,
        )

        full_df["ALDi_detailed"] = full_df.apply(
            lambda row: sum(
                [row[c] for c in full_df.columns if c.endswith("ALDi_list") and row[c]],
                [],
            ),
            axis=1,
        )
    else:
        full_df["n_valid"] = full_df.apply(
            lambda row: sum(
                [
                    row[c] == True
                    for c in row.keys()
                    if c in COUNTRIES and "Jordan" not in c and "Saudi" not in c
                ]
            ),
            axis=1,
        )

        full_df["ALDi_detailed"] = full_df.apply(
            lambda row: sum(
                [
                    row[c]
                    for c in full_df.columns
                    if c.endswith("ALDi_list")
                    and row[c]
                    and "Jordan" not in c
                    and "Saudi" not in c
                ],
                [],
            ),
            axis=1,
        )

    dev_split_sentences = pd.read_csv(
        "data/analysis/NADI2024_subtask1_dev2.tsv", sep="\t"
    )["sentence"].tolist()
    full_df["split"] = full_df["sentence"].apply(
        lambda s: "dev" if s in dev_split_sentences else "test"
    )

    return full_df[[c for c in full_df if not c.endswith("ALDi_list")]]


def normalize_text(text):
    raw_characters = [
        "أ",
        "إ",
        "آ",
        "ٱ",
        "ة",
        "ى",
        "ؤ",
        "ئ",
    ]

    normalized_characters = [
        "ا",
        "ا",
        "ا",
        "ا",
        "ه",
        "ي",
        "ء",
        "ء",
    ]

    # Remove diacritics
    p_tashkeel = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
    text = re.sub(p_tashkeel, "", text)

    # Remove punctuation
    punctuation_pattern = r"[_?؟!ـ.،'#-]|\""
    text = re.sub(punctuation_pattern, " ", text)

    # Normalize characters
    text = unicodedata.normalize("NFKC", text)

    for char, char_normalized in zip(raw_characters, normalized_characters):
        text = text.replace(char, char_normalized)

    # Remove extra spaces
    return " ".join(text.split())


def load_distinctive_words(dataset, country):
    if dataset == "TWT15DA":
        filename = f"data/Twt15DA_Lists/list_{country}_pmi.txt"
        with open(filename, "r") as f:
            lines = [l.strip() for l in f]
            for l in lines:
                assert (
                    sum([c == "," for c in l]) <= 1
                ), f"Problem with the number of commas in '{l}'"

            return sorted(set([normalize_text(l.split(",")[0].strip()) for l in lines]))

    elif dataset == "DART":
        with open(f"data/DART/tracking-phrases/{country}.txt", encoding="utf-8") as f:
            cues = sorted(
                set([normalize_text(re.sub(r"\ufeff", "", l.strip())) for l in f])
            )

        single_word_cues = [c for c in cues if " " not in c]
        multiple_word_cues = [c for c in cues if " " in c]

        return single_word_cues + multiple_word_cues

    elif dataset == "DIAL2MSA":
        with open(f"data/DIAL2MSA/{country}StrongWords.txt") as f:
            return sorted(set([normalize_text(l.strip()) for l in f]))
