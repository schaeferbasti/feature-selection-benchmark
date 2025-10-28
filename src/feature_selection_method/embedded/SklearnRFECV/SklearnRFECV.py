import pandas as pd

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

from src.utils.get_data import concat_data, get_dataset_split

def main(dataset_id, seed):
    X_train, y_train, X_test, y_test, dataset_metadata = get_dataset_split(dataset_id, seed)
    # RFECV
    print("Embedded Method: Recursive Feature Elimination with CV, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("data/embedded/SklearnRFECV_" + str(dataset_id) + ".parquet")  # ../
        print("File exists" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection")
        rfe = RFECV(estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), step=1)
        rfe.fit(X_train, y_train)
        X_train_new = rfe.transform(X_train)
        X_test_new = rfe.transform(X_test)
        selected_features = X_train.columns[rfe.get_support()]
        # Transform the data and wrap it back into DataFrames
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("data/embedded/SklearnRFECV_" + str(dataset_id) + ".parquet")  # ../
        print("File created" + str(data.head()) + "\n\n")


if __name__ == '__main__':
    dataset_id = 146820
    seed = 1
    main(dataset_id, seed)