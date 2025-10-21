import pandas as pd

from sklearn.feature_selection import VarianceThreshold

from src.utils.get_data import concat_data, get_dataset_split



def main(dataset_id):
    X_train, y_train, X_test, y_test, dataset_metadata = get_dataset_split(dataset_id)
    # Variance Threshold
    print("Filter Method: Variance Threshold, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("data/filter/SklearnVarianceThreshold_" + str(dataset_id) + ".parquet")  # ../
        print("File exists" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection")
        variance_threshold = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X_train_new = variance_threshold.fit_transform(X_train)
        X_test_new = variance_threshold.transform(X_test)
        selected_features = X_train.columns[variance_threshold.get_support()]
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("data/filter/SklearnVarianceThreshold_" + str(dataset_id) + ".parquet")  # ../
        print("File created" + str(data.head()) + "\n\n")


if __name__ == '__main__':
    dataset_id = 146820
    main(dataset_id)
