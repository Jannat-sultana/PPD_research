from sklearn.model_selection import train_test_split


def create_fixed_split(
    df,
    target="Risk",
    test_size=0.20,
    random_state=42
):
    """
    Create the single authoritative train/test split
    used throughout the entire project.
    """

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target],
        random_state=random_state
    )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


def split_data(
    df,
    target="Risk",
    test_size=0.20,
    random_state=42
):
    """
    Backward-compatible helper returning X/y splits.
    """

    train_df, test_df = create_fixed_split(
        df,
        target=target,
        test_size=test_size,
        random_state=random_state
    )

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test