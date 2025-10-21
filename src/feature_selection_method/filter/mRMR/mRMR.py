from mrmr import mrmr_classif
from mrmr import mrmr_regression
from mrmr import mrmr_base

import pandas as pd

from src.utils.get_data import concat_data, get_dataset_split


def main(dataset_id):
    X_train, y_train, X_test, y_test, dataset_metadata = get_dataset_split(dataset_id)
    # mRMR
    print("Filter Method: mRMR, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("data/filter/SklearnSelectKBestChi2_" + str(dataset_id) + ".parquet")  # ../
        print("File exists" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection")
        selected_features = mrmr_classif(X=X_train, y=y_train, K=10)
        # Transform the data and wrap it back into DataFrames
        X_train_new = pd.DataFrame(X_train, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("data/filter/mRMR" + str(dataset_id) + ".parquet")  # ../
        print("File created" + str(data.head()) + "\n\n")


if __name__ == '__main__':
    dataset_id = 146820
    main(dataset_id)
