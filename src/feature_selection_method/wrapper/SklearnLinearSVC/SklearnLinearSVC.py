import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

from src.utils.get_data import concat_data, get_dataset_split


def main(dataset_id):
    X_train, y_train, X_test, y_test, dataset_metadata = get_dataset_split(dataset_id)
    # Select From Model (Linear SVC)
    print("Wrapper Method: Linear SVC, Penalty: L1, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("data/wrapper/SklearnLinearSVC_" + str(dataset_id) + ".parquet")  # ../
        print("File exists" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection")
        try:
            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
        except ValueError:
            X_train.fillna(method="ffill", inplace=True)
            X_test.fillna(method="ffill", inplace=True)
            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
        model = SelectFromModel(lsvc, prefit=True)
        X_train_new = model.transform(X_train)
        X_test_new = model.transform(X_test)
        selected_features = X_train.columns[model.get_support()]
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("data/wrapper/SklearnLinearSVC_" + str(dataset_id) + ".parquet")  # ../
        print("File created" + str(data.head()) + "\n\n")


if __name__ == '__main__':
    dataset_id = 146820
    main(dataset_id)
