import pandas as pd

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

from src.utils.get_data import concat_data, get_dataset_split


def main(dataset_id, seed):
    X_train, y_train, X_test, y_test, dataset_metadata = get_dataset_split(dataset_id, seed)
    # Forward Sequential Feature Selector
    print("Wrapper Method: Forward Sequential Feature Selector, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("data/wrapper/SklearnForwardSFS_" + str(dataset_id) + ".parquet")  # ../
        print("File exists" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection")
        sfs = SequentialFeatureSelector(
            estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train),
            direction='forward', cv=10)
        sfs = sfs.fit(X_train, y_train)
        selected_features = X_train.columns[sfs.get_support()]
        X_train_new = X_train[selected_features]
        X_test_new = X_test[selected_features]
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("data/wrapper/SklearnForwardSFS_" + str(dataset_id) + ".parquet")  # ../
        print("File created" + str(data.head()) + "\n\n")


if __name__ == '__main__':
    dataset_id = 146820
    seed = 1
    main(dataset_id, seed)