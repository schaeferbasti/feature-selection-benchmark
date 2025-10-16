# https://github.com/IIIS-Li-Group/OpenFE

import pandas as pd
from src.feature_selection_method.wrapper.OpenFE.method import OpenFE, transform

from src.utils.get_data import concat_data, get_openml_dataset_split_and_metadata


def get_openFE_features(train_x, train_y, test_x, n_jobs, name) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    openFE = OpenFE()
    features = openFE.fit(name=name, data=train_x, label=train_y, n_jobs=n_jobs)  # generate new features
    train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs)
    return train_x, test_x


def main():
    dataset_id = 146820
    try:
        pd.read_parquet("../../../data/wrapper/OpenFE_" + str(dataset_id) + ".parquet")
    except FileNotFoundError:
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
        X_train, X_test = get_openFE_features(X_train, y_train, X_test, 1, dataset_id)
        data = concat_data(X_train, y_train, X_test, y_test, "target")
        data.to_parquet("../../../data/wrapper/OpenFE_" + str(dataset_id) + ".parquet")


if __name__ == "__main__":
    main()
