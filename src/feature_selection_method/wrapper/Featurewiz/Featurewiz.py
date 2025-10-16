# https://github.com/AutoViML/featurewiz

import pandas as pd
from src.feature_selection_method.wrapper.Featurewiz.method.featurewiz import FeatureWiz

from src.utils.get_data import get_openml_dataset_split_and_metadata, concat_data


def get_featurewiz_features(train_x, train_y, test_x) -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:
    fwiz = FeatureWiz(feature_engg='', nrows=None, transform_target=True, scalers="std",
                      category_encoders="auto", add_missing=False, verbose=0, imbalanced=False,
                      ae_options={})
    train_x, train_y = fwiz.fit_transform(train_x, train_y)
    test_x = fwiz.transform(test_x)
    return train_x, test_x


def main():
    dataset_id = 146820
    try:
        pd.read_parquet("../../../data/wrapper/Featurewiz_" + str(dataset_id) + ".parquet")
    except FileNotFoundError:
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
        X_train, X_test = get_featurewiz_features(X_train, y_train, X_test)
        data = concat_data(X_train, y_train, X_test, y_test, "target")
        data.to_parquet("../../../data/wrapper/Featurewiz_" + str(dataset_id) + ".parquet")


if __name__ == "__main__":
    main()