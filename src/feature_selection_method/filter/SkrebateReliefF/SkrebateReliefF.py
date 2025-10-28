import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF

from src.utils.get_data import concat_data, get_dataset_split


def main(dataset_id, seed):
    X_train, y_train, X_test, y_test, dataset_metadata = get_dataset_split(dataset_id, seed)
    # ReliefF
    print("Wrapper Method: ReliefF, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("data/wrapper/SkrebateReliefF_" + str(dataset_id) + ".parquet")  # ../
        print("File exists" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection")
        clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100), RandomForestClassifier(n_estimators=100))
        clf = clf.fit(X_train, y_train)
        model = SelectFromModel(clf, prefit=True)
        X_train_new = model.transform(X_train)
        X_test_new = model.transform(X_test)
        selected_features = X_train.columns[model.get_support()]
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("data/wrapper/SkrebateReliefF_" + str(dataset_id) + ".parquet")  # ../
        print("File created" + str(data.head()) + "\n\n")


if __name__ == '__main__':
    dataset_id = 146820
    seed = 1
    main(dataset_id, seed)