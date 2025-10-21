import pandas as pd

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif

from src.utils.get_data import concat_data, get_dataset_split


def main(dataset_id):
    X_train, y_train, X_test, y_test, dataset_metadata = get_dataset_split(dataset_id)
    # SelectPercentile - Mutual_info_classif
    print("Filter Method: SelectPercentile, Score Function: Mutual Info Classifier, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet(
            "data/filter/SklearnSelectPercentileMutualInfoClassif_" + str(dataset_id) + ".parquet")  # ../
        print("File exists" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection")
        selectKBest = SelectPercentile(score_func=mutual_info_classif, percentile=50)
        selectKBest.fit(X_train, y_train)
        X_train_new = selectKBest.transform(X_train)
        X_test_new = selectKBest.transform(X_test)
        selected_features = X_train.columns[selectKBest.get_support()]
        # Transform the data and wrap it back into DataFrames
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("data/filter/SklearnSelectPercentileMutualInfoClassif_" + str(dataset_id) + ".parquet")  # ../
        print("File created" + str(data.head()) + "\n\n")


if __name__ == '__main__':
    dataset_id = 146820
    main(dataset_id)
