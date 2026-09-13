"""
src/data/textify.py

Maternal Health Dataset — Three-Level Text Conversion
=======================================================

Level 1: Raw encoded feature   (Residence = 1)
        v
Level 2: Semantic concept      ("village")
        v
Level 3: Natural-language text ("The mother lives in a village.")

This is your demo textification code, adapted to:
  - fix the missing TARGET_COLUMN constant
  - read text.text_column / text.separator / text.include_target
    from configs/base.yaml instead of hardcoding them
  - drop the module-level __main__ block (paths now live in
    experiments/06_textify_dataset.py, matching how 01-05 keep
    path logic out of src/)

IMPORTANT — include_target leakage warning:
  dataframe_to_text(..., include_target=True) appends a sentence
  stating the Risk label directly into the text
  (e.g. "...classified as high risk of antenatal depression.").
  That column (`{text_column}_with_target`) must NEVER be used as
  model input — only `text_column` itself should be tokenized and
  fed to BERT/BioBERT/ClinicalBERT. Keep config text.include_target
  = false unless you have a specific, separate use for the
  target-annotated text (e.g. generating captions for inspection).
"""

from pathlib import Path

import pandas as pd


TARGET_COLUMN = "Risk"


# ==============================================================
# 1. LEVEL 1 -> LEVEL 2
#    Reverse encoding maps (inverse of src/data/clean.py)
# ==============================================================

REVERSE_MAPPINGS = {

    "Residence": {
        0: "city",
        1: "village",
    },

    "Education Level": {
        0: "primary school",
        1: "high school",
        2: "college",
        3: "university",
    },

    "Husband's education level": {
        0: "primary school",
        1: "high school",
        2: "college",
        3: "university",
    },

    "Occupation before latest pregnancy": {
        0: "housewife",
        1: "student",
        2: "teacher",
        3: "service employee",
        4: "doctor",
        5: "businessperson",
        6: "other occupation",
    },

    "Occupation After Your Latest Childbirth": {
        0: "housewife",
        1: "student",
        2: "teacher",
        3: "service employee",
        4: "doctor",
        5: "businessperson",
        6: "other occupation",
    },

    "Husband’s monthly income": {
        0: "less than 5000",
        1: "5000 to 10000",
        2: "10000 to 20000",
        3: "20000 to 30000",
        4: "more than 30000",
    },

    "Total children": {
        1: "one child",
        2: "two children",
        3: "more than two children",
    },

    "Family type": {
        0: "nuclear family",
        1: "joint family",
    },

    "Number of household members": {
        0: "2 to 5 household members",
        1: "6 to 8 household members",
        2: "9 or more household members",
    },

    "Relationship with the in-laws": {
        0: "bad",
        1: "poor",
        2: "neutral",
        3: "good",
        4: "friendly",
    },

    "Relationship with husband": {
        0: "bad",
        1: "poor",
        2: "neutral",
        3: "good",
        4: "friendly",
    },

    "Relationship with the newborn": {
        0: "bad",
        1: "neutral",
        2: "good",
        3: "very good",
    },

    "Relationship between father and newborn": {
        0: "bad",
        1: "neutral",
        2: "good",
        3: "very good",
    },

    "Feeling about motherhood": {
        0: "sad",
        1: "neutral",
        2: "happy",
    },

    "Recieved Support": {
        0: "low support",
        1: "medium support",
        2: "high support",
    },

    "Major changes or losses during pregnancy": {
        0: "no major changes or losses",
        1: "major changes or losses",
    },

    "Trust and share feelings": {
        0: "does not trust and share feelings",
        1: "trusts and shares feelings",
    },

    "Pregnancy length": {
        0: "less than 5 months",
        1: "6 months",
        2: "7 months",
        3: "8 months",
        4: "9 months",
        5: "10 months",
    },

    "Pregnancy plan": {
        0: "unplanned",
        1: "planned",
    },

    "Regular checkups": {
        0: "does not attend regular checkups",
        1: "attends regular checkups",
    },

    "Fear of pregnancy": {
        0: "no fear of pregnancy",
        1: "fear of pregnancy",
    },

    "Age of newborn": {
        0: "0 to 6 months",
        1: "6 months to 1 year",
        2: "1 year to 1.5 years",
        3: "older than 1.5 years",
    },

    "Birth compliancy": {
        0: "no birth complications",
        1: "birth complications",
    },

    "Breastfeed": {
        0: "does not breastfeed",
        1: "breastfeeds",
    },

    "Newborn illness": {
        0: "no newborn illness",
        1: "newborn illness",
    },

    "Worry about newborn": {
        0: "no worry about the newborn",
        1: "worry about the newborn",
    },

    "Relax/sleep when newborn is tended": {
        0: "cannot relax or sleep when someone tends to the newborn",
        1: "can relax or sleep when someone tends to the newborn",
    },

    "Relax/sleep when the newborn is asleep": {
        0: "cannot relax or sleep when the newborn is asleep",
        1: "can relax or sleep when the newborn is asleep",
    },

    "Angry after latest child birth": {
        0: "not angry after the latest childbirth",
        1: "angry after the latest childbirth",
    },
}


# ==============================================================
# 2. LEVEL 2 -> LEVEL 3
#    Natural-language templates
# ==============================================================

TEXT_TEMPLATES = {

    "Residence":
        "The mother lives in a {concept}.",

    "Education Level":
        "The mother has completed {concept} education.",

    "Occupation before latest pregnancy":
        "Before her latest pregnancy, the mother worked as a {concept}.",

    "Occupation After Your Latest Childbirth":
        "After her latest childbirth, the mother works as a {concept}.",

    "Husband's education level":
        "The husband's education level is {concept}.",

    "Husband’s monthly income":
        "The husband's monthly income is {concept}.",

    "Total children":
        "The mother has {concept}.",

    "Family type":
        "The mother lives in a {concept}.",

    "Number of household members":
        "There are {concept}.",

    "Relationship with the in-laws":
        "The mother's relationship with her in-laws is {concept}.",

    "Relationship with husband":
        "The mother's relationship with her husband is {concept}.",

    "Relationship with the newborn":
        "The mother's relationship with the newborn is {concept}.",

    "Relationship between father and newborn":
        "The father's relationship with the newborn is {concept}.",

    "Feeling about motherhood":
        "The mother feels {concept} about motherhood.",

    "Recieved Support":
        "The mother receives {concept}.",

    "Major changes or losses during pregnancy":
        "During pregnancy, the mother experienced {concept}.",

    "Trust and share feelings":
        "The mother {concept}.",

    "Pregnancy length":
        "The pregnancy lasted {concept}.",

    "Pregnancy plan":
        "The pregnancy was {concept}.",

    "Regular checkups":
        "The mother {concept}.",

    "Fear of pregnancy":
        "The mother reports {concept}.",

    "Age of newborn":
        "The newborn is {concept} old.",

    "Birth compliancy":
        "The mother experienced {concept} during childbirth.",

    "Breastfeed":
        "The mother {concept} the newborn.",

    "Newborn illness":
        "The newborn has {concept}.",

    "Worry about newborn":
        "The mother reports {concept}.",

    "Relax/sleep when newborn is tended":
        "The mother {concept}.",

    "Relax/sleep when the newborn is asleep":
        "The mother {concept}.",

    "Angry after latest child birth":
        "The mother is {concept}.",
}


# ==============================================================
# 3. Target mapping
# ==============================================================

TARGET_MAPPING = {
    0: "low risk of antenatal depression",
    1: "high risk of antenatal depression",
}

TARGET_TEMPLATE = (
    "Based on the EPDS threshold, the mother is classified as "
    "{concept}."
)


# ==============================================================
# 4. Level 1 -> Level 2
# ==============================================================

def reverse_encode_value(feature, value):

    if feature not in REVERSE_MAPPINGS:
        raise KeyError(
            f"No reverse mapping found for feature: '{feature}'"
        )

    if pd.isna(value):
        return "unknown"

    try:
        value = int(value)
    except (ValueError, TypeError):
        pass

    mapping = REVERSE_MAPPINGS[feature]

    if value not in mapping:
        raise ValueError(
            f"Unknown encoded value '{value}' for feature "
            f"'{feature}'. Expected values: {list(mapping.keys())}"
        )

    return mapping[value]


# ==============================================================
# 5. Level 2 -> Level 3
# ==============================================================

def semantic_to_text(feature, concept):

    if feature not in TEXT_TEMPLATES:
        raise KeyError(
            f"No natural-language template found for "
            f"feature: '{feature}'"
        )

    return TEXT_TEMPLATES[feature].format(concept=concept)


# ==============================================================
# 6. Complete three-level conversion of one value
# ==============================================================

def convert_feature(feature, value):

    concept = reverse_encode_value(feature, value)
    text = semantic_to_text(feature, concept)

    return {
        "feature": feature,
        "raw_value": value,
        "semantic_concept": concept,
        "text": text,
    }


# ==============================================================
# 7. Convert one dataframe row
# ==============================================================

def row_to_three_level_text(row, feature_columns=None, separator=" "):
    """
    Only features present in REVERSE_MAPPINGS (and in `row`) are
    converted, so this works whether you pass the FULL cleaned
    row or a reduced (e.g. COMBINED feature-set) row.
    """

    if feature_columns is None:
        feature_columns = [
            feature
            for feature in REVERSE_MAPPINGS
            if feature in row.index
        ]

    sentences = []

    for feature in feature_columns:

        value = row[feature]

        if pd.isna(value):
            continue

        try:
            concept = reverse_encode_value(feature, value)
            sentence = semantic_to_text(feature, concept)
            sentences.append(sentence)

        except (KeyError, ValueError) as error:
            print(f"WARNING: {error}")

    return separator.join(sentences)


# ==============================================================
# 8. Convert target separately
# ==============================================================

def target_to_text(value):

    if pd.isna(value):
        return None

    value = int(value)

    if value not in TARGET_MAPPING:
        raise ValueError(
            f"Unknown Risk value: {value}. Expected 0 or 1."
        )

    concept = TARGET_MAPPING[value]

    return TARGET_TEMPLATE.format(concept=concept)


# ==============================================================
# 9. Create three-level dataframe (for inspection)
# ==============================================================

def create_three_level_dataframe(df, feature_columns=None):

    if feature_columns is None:
        feature_columns = [
            feature
            for feature in REVERSE_MAPPINGS
            if feature in df.columns
        ]

    records = []

    for _, row in df.iterrows():

        record = {}

        for feature in feature_columns:

            value = row[feature]

            record[f"{feature}__raw"] = value

            if pd.isna(value):
                record[f"{feature}__concept"] = None
                record[f"{feature}__text"] = None
                continue

            try:
                concept = reverse_encode_value(feature, value)
                text = semantic_to_text(feature, concept)

                record[f"{feature}__concept"] = concept
                record[f"{feature}__text"] = text

            except (KeyError, ValueError):
                record[f"{feature}__concept"] = None
                record[f"{feature}__text"] = None

        if TARGET_COLUMN in row.index:
            record[TARGET_COLUMN] = row[TARGET_COLUMN]
            record[f"{TARGET_COLUMN}_text"] = target_to_text(
                row[TARGET_COLUMN]
            )

        records.append(record)

    return pd.DataFrame(records)


# ==============================================================
# 10. Convert complete dataframe to text
# ==============================================================

def dataframe_to_text(
    df,
    output_column="text",
    feature_columns=None,
    include_target=False,
    separator=" "
):
    """
    See the leakage warning in the module docstring before
    setting include_target=True.
    """

    df = df.copy()

    if feature_columns is None:
        feature_columns = [
            feature
            for feature in REVERSE_MAPPINGS
            if feature in df.columns
        ]

    df[output_column] = df.apply(
        lambda row: row_to_three_level_text(
            row,
            feature_columns,
            separator=separator
        ),
        axis=1
    )

    if include_target and TARGET_COLUMN in df.columns:

        df[f"{output_column}_with_target"] = df.apply(
            lambda row: separator.join([
                row[output_column],
                target_to_text(row[TARGET_COLUMN]) or ""
            ]),
            axis=1
        )

    return df


# ==============================================================
# 11. Save three-level dataset (raw + concept + text, for QA)
# ==============================================================

def save_three_level_dataset(df, output_path, feature_columns=None):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_df = create_three_level_dataframe(df, feature_columns)
    converted_df.to_csv(output_path, index=False)

    print("=" * 70)
    print("THREE-LEVEL DATASET CREATED")
    print("=" * 70)
    print(f"Output : {output_path}")
    print(f"Rows   : {len(converted_df)}")
    print(f"Columns: {len(converted_df.columns)}")
    print("=" * 70)

    return converted_df


# ==============================================================
# 12. Save final NLP dataset
# ==============================================================

def save_text_dataset(
    df,
    output_path,
    feature_columns=None,
    include_target=False,
    text_column="text",
    separator=" "
):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_df = dataframe_to_text(
        df,
        output_column=text_column,
        feature_columns=feature_columns,
        include_target=include_target,
        separator=separator
    )

    converted_df.to_csv(output_path, index=False)

    print("=" * 70)
    print("NATURAL-LANGUAGE DATASET CREATED")
    print("=" * 70)
    print(f"Output     : {output_path}")
    print(f"Rows       : {len(converted_df)}")
    print(f"Text column: {text_column}")
    print("=" * 70)

    return converted_df