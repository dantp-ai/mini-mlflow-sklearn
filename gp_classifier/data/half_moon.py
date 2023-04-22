import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split


def get_half_moon_data(
    data_path, data_repo, data_version, split="all", test_size=0.3, seed=None
):
    """
    split (str): which split of the data to return (default: 'all')
        'all': train and test splits
        'test': test split only
        'train': train split only
    """
    # Read data from dvc
    data_url = dvc.api.get_url(path=data_path, repo=data_repo, rev=data_version)
    data_url = data_url.strip("/")
    df = pd.read_csv(data_url, sep=",")

    # Split data into features and targets
    features = [x for x in list(df.columns) if x != "label"]
    X_raw = df[features]
    y_raw = df["label_0"]

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=y_raw,
    )

    if split == "train":
        return X_train, y_train
    elif split == "test":
        return X_test, y_test
    elif split == "all":
        return X_train, X_test, y_train, y_test
    else:
        raise ValueError("split can be one of `train`, `test`, or `all`.")
