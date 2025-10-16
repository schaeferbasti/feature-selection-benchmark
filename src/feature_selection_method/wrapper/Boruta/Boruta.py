# https://github.com/scikit-learn-contrib/boruta_py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.feature_selection_method.wrapper.Boruta.method import BorutaPy
from src.utils.get_data import get_openml_dataset_split_and_metadata, concat_data, preprocess_data


def get_boruta_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators="auto", verbose=2)

    train_x, test_x = preprocess_data(train_x, test_x)

    for column in train_x.select_dtypes(include=['object', 'category']).columns:
        train_x[column], uniques = pd.factorize(train_x[column])
    for column in test_x.select_dtypes(include=['object', 'category']).columns:
        test_x[column], uniques = pd.factorize(test_x[column])
    train_y = pd.DataFrame(train_y)
    for column in train_y.select_dtypes(include=['object', 'category']).columns:
        train_y[column], uniques = pd.factorize(train_y[column])

    train_x_np = train_x.values
    train_y_np = train_y.values
    test_x_np = test_x.values

    feat_selector.fit(train_x_np, train_y_np.ravel())

    # Transform the training and testing data
    train_x_selected = feat_selector.transform(train_x_np)
    test_x_selected = feat_selector.transform(test_x_np)

    # Convert the transformed data back to DataFrame
    train_x_selected_df = pd.DataFrame(train_x_selected, columns=train_x.columns[feat_selector.support_])
    test_x_selected_df = pd.DataFrame(test_x_selected, columns=test_x.columns[feat_selector.support_])

    print(train_x_selected_df.columns)
    print(test_x_selected_df.columns)
    return train_x_selected_df, test_x_selected_df


def main():
    dataset_id = 146820
    try:
        pd.read_parquet("../../../data/wrapper/Boruta_" + str(dataset_id) + ".parquet")
    except FileNotFoundError:
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
        X_train, X_test = get_boruta_features(X_train, y_train, X_test)
        data = concat_data(X_train, y_train, X_test, y_test, "target")
        data.to_parquet("../../../data/wrapper/Boruta_" + str(dataset_id) + ".parquet")


if __name__ == "__main__":
    main()
