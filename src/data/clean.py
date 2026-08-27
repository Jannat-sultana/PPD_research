import pandas as pd


TARGET = "Risk"


def load_raw_data(path):
    """Load the original dataset."""
    return pd.read_csv(path)


def clean_dataset(df):

    # ==================================================
    # 1. Create target variable from EPDS Score
    # ==================================================

    # EPDS Score >= 13  → High Risk (1)
    # EPDS Score < 13   → Low Risk (0)

    df[TARGET] = df["EPDS Score"].apply(
        lambda x: 1 if x >= 13 else 0
    )

    # ==================================================
    # 2. Remove unnecessary columns
    # ==================================================

    columns_to_drop = [
        "sr",
        "Age",
        "Number of the latest pregnancy",
        "Gender of newborn",
        "Mode of delivery",
        "Marital status",
        "Abuse",
        "Depression before pregnancy (PHQ2)",
        "Depression during pregnancy (PHQ2)",
        "PHQ9 Result",
        "PHQ9 Score",
        "EPDS Result",
        "EPDS Score"
    ]

    df = df.drop(columns=columns_to_drop)

    # ===============================
    # 3. Remove null columns 
    # ===============================

    columns_to_drop = [
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

    df = df.drop(columns=columns_to_drop)
    df.columns = df.columns.str.strip()
    # ==================================================
    # 4. Fill missing categorical values
    # ==================================================

    mode_columns = [
        "Education Level",
        "Husband's education level",
        "Husband’s monthly income",
        "Trust and share feelings"
    ]

    for column in mode_columns:

        if column in df.columns:
            df[column] = df[column].fillna(
                df[column].mode()[0]
            )


    # ==================================================
    # 5. Standardize category names
    # ==================================================

    rename_map = {
        "High school": "high school",
        "Primary school": "primary school",
        "House wife": "housewife",
        "More than Two": "more than two"
    }

    columns_to_standardize = [
        "Occupation before latest pregnancy",
        "Occupation After Your Latest Childbirth",
        "Total children",
        "Education Level",
        "Husband's education level"
    ]

    for column in columns_to_standardize:

        if column in df.columns:
            df[column] = df[column].replace(rename_map)

    # ==================================================
    # 6. Education categories
    # ==================================================

    education_map = {
        "University": "university",
        "College": "college",
        "High School": "high school",
        "Primary School": "primary school"
    }

    df["Education Level"] = \
        df["Education Level"].replace(education_map)

    df["Husband's education level"] = \
        df["Husband's education level"].replace(education_map)

    # ==================================================
    # 7. Occupation
    # ==================================================

    occupation_map = {
        "Housewife": "housewife",
        "Student": "student",
        "Teacher": "teacher",
        "Service": "service",
        "Doctor": "doctor",
        "Business": "business",
        "Other": "other"
    }

    df["Occupation before latest pregnancy"] = \
        df["Occupation before latest pregnancy"].replace(
            occupation_map
        )

    # ==================================================
    # 8. Encode Total children
    # ==================================================

    children_map = {
        "One": 1,
        "Two": 2,
	    "More than two": 3
    }

    df["Total children"] = \
        df["Total children"].map(children_map)

    # ==================================================
    # 9. Binary encoding
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

    df[binary_columns] = (df[binary_columns].replace(binary_map).infer_objects(copy=False)
)

    # ==================================================
    # 10. Residence
    # ==================================================

    residence_map = {
        "City": 0,
        "Village": 1
    }

    df["Residence"] = \
        df["Residence"].map(residence_map)

    # ==================================================
    # 11. Education
    # ==================================================

    education_binary_map = {
        "University": 0,
        "Pre-university": 1
    }

    df["Education Level"] = \
        df["Education Level"].map(
            education_binary_map
        )

    df["Husband's education level"] = \
        df["Husband's education level"].map(
            education_binary_map
        )

    # ==================================================
    # 12. Husband's income
    # ==================================================

    income_map = {
        "Less than 5000": 0,
        "5000 to 10000": 1,
        "10000 to 20000": 2,
	    "20000 to 30000": 3,
	    "More than 30000": 4
    }

    df["Husband’s monthly income"] = \
        df["Husband’s monthly income"].map(
            income_map
        )

    # ==================================================
    # 13. Occupation
    # ==================================================

    occupation_encoding = {
        "housewife": 0,
        "student": 1,
        "teacher": 2,
        "service": 3,
        "doctor": 4,
        "business": 5,
        "other": 6
    }

    df["Occupation before latest pregnancy"] = \
        df[
            "Occupation before latest pregnancy"
        ].map(occupation_encoding)

    df["Occupation After Your Latest Childbirth"] = \
        df[
            "Occupation before latest pregnancy"
        ].map(occupation_encoding)

    # ==================================================
    # 14. Family type
    # ==================================================

    family_map = {
        "Nuclear": 0,
        "Joint": 1
    }

    df["Family type"] = \
        df["Family type"].map(family_map)

    # ==================================================
    # 15. Household members
    # ==================================================

    household_map = {
        "2 to 5": 0,
        "6 to 8": 1,
	    "9 or more": 2
    }

    df["Number of household members"] = \
        df[
            "Number of household members"
        ].map(household_map)

    # ==================================================
    # 16. Relationships
    # ==================================================

    relationship_map = {
        "Bad": 0,
        "Poor": 1,
        "Neutral": 2,
	    "Good": 3,
	    "Friendly": 4
    }

    df["Relationship with the in-laws"] = \
        df[
            "Relationship with the in-laws"
        ].map(relationship_map)

    df["Relationship with husband"] = \
        df[
            "Relationship with husband"
        ].map(relationship_map)

    # ==================================================
    # 17. Feeling about motherhood
    # ==================================================

    motherhood_map = {
        "Sad": 0,
        "Neutral": 1,
        "Happy": 2
    }

    df["Feeling about motherhood"] = \
        df[
            "Feeling about motherhood"
        ].map(motherhood_map)

    # ==================================================
    # 18. Received support
    # ==================================================

    support_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    df["Recieved Support"] = \
        df["Recieved Support"].map(
            support_map
        )

    # ==================================================
    # 19. Pregnancy length
    # ==================================================

    pregnancy_length_map = {
        "Less than 5 months": 0,
        "6 months": 1,
	    "7 months": 2,
	    "8 months": 3,
        "9 months": 4,
	    "10 months": 5
    }

    df["Pregnancy length"] = \
        df[
            "Pregnancy length"
        ].map(pregnancy_length_map)


    # ==================================================
    # 21. Relationship with newborn
    # ==================================================

    newborn_relationship_map = {
        "Bad": 0,
        "Neutral": 1,
        "Good": 2,
        "Very good": 3
    }

    df["Relationship with the newborn"] = \
        df["Relationship with the newborn"].map(
            newborn_relationship_map
        )


    # ==================================================
    # 22. Relationship between father and newborn
    # ==================================================

    father_newborn_relationship_map = {
        "Bad": 0,
        "Neutral": 1,
        "Good": 2,
        "Very good": 3
    }

    df["Relationship between father and newborn"] = \
        df["Relationship between father and newborn"].map(
            father_newborn_relationship_map
        )


    # ==================================================
    # 23. Age of newborn
    # ==================================================

    newborn_age_map = {
        "0 to 6 months": 0,
        "6 months to 1 year": 1,
        "1 year to 1.5 year": 2,
        "Older than 1.5 year": 3
    }

    df["Age of newborn"] = \
        df["Age of newborn"].map(
            newborn_age_map
        )


    # ==================================================
    # 24. Birth compliancy
    # ==================================================

    yes_no_map = {
        "Yes": 1,
        "No": 0
    }

    binary_columns_after_birth = [
        "Birth compliancy",
        "Breastfeed",
        "Newborn illness",
        "Worry about newborn",
        "Relax/sleep when newborn is tended",
        "Relax/sleep when the newborn is asleep",
        "Angry after latest child birth"
    ]

    for column in binary_columns_after_birth:

        if column in df.columns:
            df[column] = df[column].map(yes_no_map)


    return df