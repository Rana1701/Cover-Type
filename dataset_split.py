import pandas as pd


def create_fixed_split(
    dataset,
    label_column="Cover_Type",
    test_samples_per_class=500,
    random_state=42,
):
    test_parts = []
    train_parts = []

    for label, group in dataset.groupby(label_column):
        test_part = group.sample(n=test_samples_per_class, random_state=random_state)
        train_part = group.drop(test_part.index)

        test_parts.append(test_part)
        train_parts.append(train_part)

    test_dataset = pd.concat(test_parts).sample(
        frac=1, random_state=random_state + 10_000
    ).reset_index(drop=True)
    train_dataset = pd.concat(train_parts).sample(
        frac=1, random_state=random_state + 20_000
    ).reset_index(drop=True)

    x_train = train_dataset.drop(label_column, axis=1)
    y_train = train_dataset[label_column]

    x_test = test_dataset.drop(label_column, axis=1)
    y_test = test_dataset[label_column]

    return x_train, x_test, y_train, y_test
