# https://github.com/Doctorado-ML/mufs

import pandas as pd

from src.feature_selection_method.filter.CFS.MUFS import MUFS
from src.utils.get_data import concat_data, get_openml_dataset_split_and_metadata, preprocess_data


def get_correlationbased_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    for column in train_x.select_dtypes(include=['object', 'category']).columns:
        train_x[column], uniques = pd.factorize(train_x[column])
    for column in test_x.select_dtypes(include=['object', 'category']).columns:
        test_x[column], uniques = pd.factorize(test_x[column])
    train_x, test_x = preprocess_data(train_x, test_x)
    train_y = pd.DataFrame(train_y)
    for column in train_y.select_dtypes(include=['object', 'category']).columns:
        train_y[column], uniques = pd.factorize(train_y[column])
    float_array_of_features = train_x.to_numpy().astype("float64")
    float_array_of_targets = train_y.to_numpy().astype("float64")
    float_array_of_targets = float_array_of_targets[:, 0]

    mufs = MUFS(discrete=False)
    cfs_f = mufs.cfs(float_array_of_features, float_array_of_targets).get_results()
    fcbf_f = mufs.fcbf(float_array_of_features, float_array_of_targets, 1e-3).get_results()
    print(cfs_f)
    print(mufs.get_scores())

    train_x = train_x.iloc[:, cfs_f]
    test_x = test_x.iloc[:, cfs_f]

    return train_x, test_x


def main():
    dataset_id = 146820
    try:
        pd.read_parquet("../../../data/filter/CFS_" + str(dataset_id) + ".parquet")
    except FileNotFoundError:
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
        X_train, X_test = get_correlationbased_features(X_train, y_train, X_test)
        data = concat_data(X_train, y_train, X_test, y_test, "target")
        data.to_parquet("../../../data/filter/CFS_" + str(dataset_id) + ".parquet")


if __name__ == "__main__":
    main()
