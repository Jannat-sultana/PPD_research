import pandas as pd


TARGET = "Risk"


def load_raw_data(path):
    """
    Load the original raw dataset.
    """
    return pd.read_csv(path)


def clean_dataset(df):
    """
    Clean the raw maternal mental health dataset.

    Target definition:
        EPDS Result >= 13 -> Risk = 1 (High)
        EPDS Result < 13  -> Risk = 0 (Low)

    Important:
        EPDS Result itself is removed from the final features
        to prevent target leakage.
    """

    df = df.copy()

    # ==================================================
    # 1. Create target BEFORE dropping EPDS Result
    # ==================================================

    df[TARGET] = (
        pd.to_numeric(
            df["EPDS Result"],
            errors="coerce"
        ) >= 13
    ).astype(int)

    # ==================================================
    # 2. Remove columns that should not be features
    # ==================================================

    columns_to_drop = [
        "sr",
        "Age",
        "Number of the latest pregnancy",
        "Gender of newborn",
        "Mode of delivery",
        "Marital status",
        "Abuse",

        # Depression-related variables
        "Depression before pregnancy (PHQ2)",
        "Depression during pregnancy (PHQ2)",
        "PHQ9 Result",
        "PHQ9 Score",
        "EPDS Result",

        # Variables with null / unusable information
        "Monthly income before latest pregnancy",
        "Current monthly income",
        "Addiction",
        "Disease before pregnancy",
        "History of pregnancy loss",
        "Diseases during pregnancy",
        "Need for Support",
        "Age of immediate older children",
        "Feeling for regular activities"
    ]

    # Only drop columns that actually exist
    columns_to_drop = [
        col for col in columns_to_drop
        if col in df.columns
    ]

    df = df.drop(columns=columns_to_drop)

    # ==================================================
    # 3. Fill missing categorical values
    # ==================================================

    mode_columns = [
        "Education Level",
        "Husband's education level",
        "Husband’s monthly income",
        "Trust and share feelings"
    ]

    for column in mode_columns:

        if column in df.columns:

            mode = df[column].mode()

            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])

    # ==================================================
    # 4. Standardize category names
    # ==================================================

    rename_map = {
        "High school": "High School",
        "Primary school": "Primary School",
        "House wife": "Housewife",
        "More than Two": "More than two"
    }

    columns_to_standardize = [
        "Occupation before latest pregnancy",
        "Total children",
        "Education Level",
        "Husband's education level"
    ]

    for column in columns_to_standardize:

        if column in df.columns:
            df[column] = df[column].replace(rename_map)

    # ==================================================
    # 5. Education categories
    # ==================================================

    education_map = {
        "University": "University",
        "College": "Pre-university",
        "High School": "Pre-university",
        "Primary School": "Pre-university"
    }

    if "Education Level" in df.columns:
        df["Education Level"] = (
            df["Education Level"].replace(education_map)
        )

    if "Husband's education level" in df.columns:
        df["Husband's education level"] = (
            df["Husband's education level"].replace(
                education_map
            )
        )

    # ==================================================
    # 6. Occupation
    # ==================================================

    occupation_map = {
        "Housewife": "Housewife",
        "Student": "Student",
        "Teacher": "Working",
        "Service": "Working",
        "Doctor": "Working",
        "Business": "Working",
        "Other": "Working"
    }

    if "Occupation before latest pregnancy" in df.columns:

        df["Occupation before latest pregnancy"] = (
            df["Occupation before latest pregnancy"]
            .replace(occupation_map)
        )

    # ==================================================
    # 7. Total children
    # ==================================================

    children_map = {
        "One": 1,
        "Two": 2,
        "More than two": 3
    }

    if "Total children" in df.columns:

        df["Total children"] = (
            df["Total children"].map(children_map)
        )

    # ==================================================
    # 8. Binary encoding
    # ==================================================

    binary_map = {
        "Yes": 1,
        "No": 0,
        "Male": 1,
        "Female": 0
    }

    binary_columns = [
        "Major changes or losses during pregnancy",
        "Pregnancy plan",
        "Regular checkups",
        "Fear of pregnancy",
        "Trust and share feelings"
    ]

    existing_binary_columns = [
        col for col in binary_columns
        if col in df.columns
    ]

    if existing_binary_columns:

        df[existing_binary_columns] = (
            df[existing_binary_columns]
            .replace(binary_map)
            .infer_objects(copy=False)
        )

    # ==================================================
    # 9. Residence
    # ==================================================

    residence_map = {
        "City": 0,
        "Village": 1
    }

    if "Residence" in df.columns:

        df["Residence"] = (
            df["Residence"].map(residence_map)
        )

    # ==================================================
    # 10. Education
    # ==================================================

    education_binary_map = {
        "University": 0,
        "Pre-university": 1
    }

    if "Education Level" in df.columns:

        df["Education Level"] = (
            df["Education Level"]
            .map(education_binary_map)
        )

    if "Husband's education level" in df.columns:

        df["Husband's education level"] = (
            df["Husband's education level"]
            .map(education_binary_map)
        )

    # ==================================================
    # 11. Husband's income
    # ==================================================

    income_map = {
        "Less than 5000": 0,
        "5000 to 10000": 1,
        "10000 to 20000": 2,
        "20000 to 30000": 3,
        "More than 30000": 4
    }

    income_column = "Husband’s monthly income"

    if income_column in df.columns:

        df[income_column] = (
            df[income_column].map(income_map)
        )

    # ==================================================
    # 12. Occupation encoding
    # ==================================================

    occupation_encoding = {
        "Housewife": 0,
        "Student": 1,
        "Working": 2
    }

    occupation_column = (
        "Occupation before latest pregnancy"
    )

    if occupation_column in df.columns:

        df[occupation_column] = (
            df[occupation_column]
            .map(occupation_encoding)
        )

    # ==================================================
    # 13. Family type
    # ==================================================

    family_map = {
        "Nuclear": 0,
        "Joint": 1
    }

    if "Family type" in df.columns:

        df["Family type"] = (
            df["Family type"].map(family_map)
        )

    # ==================================================
    # 14. Household members
    # ==================================================

    household_map = {
        "2 to 5": 0,
        "6 to 8": 1,
        "9 or more": 2
    }

    household_column = (
        "Number of household members"
    )

    if household_column in df.columns:

        df[household_column] = (
            df[household_column]
            .map(household_map)
        )

    # ==================================================
    # 15. Relationships
    # ==================================================

    relationship_map = {
        "Bad": 0,
        "Poor": 1,
        "Neutral": 2,
        "Good": 3,
        "Friendly": 4
    }

    for column in [
        "Relationship with the in-laws",
        "Relationship with husband"
    ]:

        if column in df.columns:

            df[column] = (
                df[column].map(relationship_map)
            )

    # ==================================================
    # 16. Feeling about motherhood
    # ==================================================

    motherhood_map = {
        "Sad": 0,
        "Neutral": 1,
        "Happy": 2
    }

    if "Feeling about motherhood" in df.columns:

        df["Feeling about motherhood"] = (
            df["Feeling about motherhood"]
            .map(motherhood_map)
        )

    # ==================================================
    # 17. Received support
    # ==================================================

    support_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    support_column = "Recieved Support"

    if support_column in df.columns:

        df[support_column] = (
            df[support_column].map(support_map)
        )

    # ==================================================
    # 18. Pregnancy length
    # ==================================================

    pregnancy_length_map = {
        "Less than 5 months": 0,
        "6 months": 1,
        "7 months": 2,
        "8 months": 3,
        "9 months": 4,
        "10 months": 5
    }

    if "Pregnancy length" in df.columns:

        df["Pregnancy length"] = (
            df["Pregnancy length"]
            .map(pregnancy_length_map)
        )

    # ==================================================
    # 19. Final validation
    # ==================================================

    if TARGET not in df.columns:
        raise ValueError(
            "Target column 'Risk' was not created."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Risk contains missing values."
        )

    # Make sure target is integer
    df[TARGET] = df[TARGET].astype(int)

    # Check for remaining non-numeric columns
    non_numeric = df.drop(
        columns=[TARGET]
    ).select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if non_numeric:

        raise ValueError(
            "The following columns are still non-numeric: "
            f"{non_numeric}"
        )

    return df