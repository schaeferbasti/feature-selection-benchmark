from __future__ import annotations

import multiprocessing
import time
from functools import partial

import pandas as pd
import psutil

from Add_Pandas_Metafeatures import add_pandas_metadata_columns
from autogluon.tabular.models import CatBoostModel

from src.utils.create_feature_and_featurename import create_featurenames, extract_operation_and_original_features, create_feature_and_featurename
from src.utils.get_data import get_openml_dataset_split_and_metadata, concat_data
from src.utils.get_matrix import get_matrix_core_columns
from multiprocessing import Value
import ctypes

import warnings
warnings.filterwarnings('ignore')

last_reset_time = Value(ctypes.c_double, time.time())
merge_keys = ["dataset - id", "feature - name", "operator", "model", "improvement"]

def safe_merge(left, right):
    return pd.merge(left, right, on=merge_keys, how="inner")


def get_additional_features(X, y, prediction_result):
    additional_feature_list = prediction_result['feature - name']
    for additional_feature in additional_feature_list:
        operation, original_features = extract_operation_and_original_features(additional_feature)
        if len(original_features) == 2:
            feature, featurename = create_feature_and_featurename(X[original_features[0]], X[original_features[1]], operation)
        else:
            feature, featurename = create_feature_and_featurename(X[original_features[0]], None, operation)
        if feature is not None:
            feature = pd.Series(feature).to_frame(additional_feature)
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            feature = feature.reset_index(drop=True)
            X = pd.concat([X, feature], axis=1)
        else:
            X = X.drop(featurename, axis=1)
    return X, y


def execute_feature_engineering_recursive(prediction_result, X, y):
    dataset_id = int(prediction_result["dataset - id"].values[0])
    model = prediction_result["model"].values[0]
    X, y = get_additional_features(X, y, prediction_result)
    return X, y, dataset_id, model


def create_empty_core_matrix_for_dataset(X_train, model, dataset_id) -> pd.DataFrame:
    columns = get_matrix_core_columns()
    comparison_result_matrix = pd.DataFrame(columns=columns)
    for feature1 in X_train.columns:
        featurename = "without - " + str(feature1)
        columns = get_matrix_core_columns()
        new_rows = pd.DataFrame(columns=columns)
        operator = "delete"
        new_rows.loc[len(new_rows)] = [
            dataset_id,
            featurename,
            operator,
            model,
            0
        ]
        comparison_result_matrix = pd.concat([comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
    columns = get_matrix_core_columns()
    new_rows = pd.DataFrame(columns=columns)
    featurenames = create_featurenames(X_train.columns)
    for i in range(len(featurenames)):
        operator, _ = extract_operation_and_original_features(featurenames[i])
        new_rows.loc[len(new_rows)] = [
            dataset_id,
            featurenames[i],
            operator,
            model,
            0
        ]
    comparison_result_matrix = pd.concat([comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True)
    return comparison_result_matrix


def recursive_feature_addition(X, y, X_test, y_test, model, dataset_metadata, category_to_drop, wanted_min_relative_improvement, dataset_id):
    result_matrix = pd.read_parquet("Pandas_Matrix_Complete.parquet")
    datasets = pd.unique(result_matrix["dataset - id"]).tolist()
    if dataset_id in datasets:
        result_matrix = result_matrix[result_matrix["dataset - id"] != dataset_id]
    comparison_result_matrix = create_empty_core_matrix_for_dataset(X, model, dataset_id)
    comparison_result_matrix = add_pandas_metadata_columns(dataset_metadata, X, comparison_result_matrix)
    # Predict and split again
    X_new, y_new = predict_improvement(result_matrix, comparison_result_matrix, X, y, wanted_min_relative_improvement)
    if X_new.equals(X):  # if X_new.shape == X.shape
        try:
            y_list = y['target'].tolist()
            y_series = pd.Series(y_list)
            y = y_series
        except KeyError:
            print("")
        data = concat_data(X, y, X_test, y_test, "target")
        data.to_parquet("../../data/metalearning/MetaFE_" + str(dataset_id) + ".parquet")
        return X, y
    else:
        try:
            y_list = y_new['target'].tolist()
            y_series = pd.Series(y_list)
            y_new = y_series
        except KeyError:
            print("")
        data = concat_data(X_new, y_new, X_test, y_test, "target")
        data.to_parquet("MetaFE_" + str(dataset_id) + ".parquet")
        return recursive_feature_addition(X_new, y_new, X_test, y_test, model, dataset_metadata, category_to_drop, wanted_min_relative_improvement, dataset_id)


def predict_improvement(result_matrix, comparison_result_matrix, X_train, y_train, wanted_min_relative_improvement):
    y_result = result_matrix["improvement"]
    result_matrix = result_matrix.drop("improvement", axis=1)
    comparison_result_matrix = comparison_result_matrix.drop("improvement", axis=1)
    clf = CatBoostModel()
    clf.fit(X=result_matrix, y=y_result)
    # Predict and score
    comparison_result_matrix.columns = comparison_result_matrix.columns.astype(str)
    comparison_result_matrix = comparison_result_matrix[result_matrix.columns]
    prediction = clf.predict(X=comparison_result_matrix)
    prediction_df = pd.DataFrame(prediction, columns=["predicted_improvement"])
    prediction_concat_df = pd.concat([comparison_result_matrix[["dataset - id", "feature - name", "model"]], prediction_df], axis=1)
    best_operation = prediction_concat_df.nlargest(n=1, columns="predicted_improvement", keep="first")
    if best_operation["predicted_improvement"].values[0] < wanted_min_relative_improvement:
        print("Predicted improvement of best operation: " + str(best_operation["predicted_improvement"].values[0]) + " - not good enough")
        return X_train, y_train
    else:
        print("Predicted improvement of best operation: " + str(best_operation["predicted_improvement"].values[0]) + " - execute feature engineering")
        X, y, _, _ = execute_feature_engineering_recursive(best_operation, X_train, y_train)
    return X, y


def process_method(dataset_id, model, wanted_min_relative_improvement):
    last_reset_time.value = time.time()
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
    try:
        pd.read_parquet("../../data/metalearning/MetaFE_" + str(dataset_id) + ".parquet")
    except FileNotFoundError:
        X_train, y_train = recursive_feature_addition(X_train, y_train, X_test, y_test, model, dataset_metadata, None, wanted_min_relative_improvement, dataset_id)
        data = concat_data(X_train, y_train, X_test, y_test, "target")
        data.to_parquet("../../data/metalearning/MetaFE_" + str(dataset_id) + ".parquet")


def run_process_method(dataset_id, model, improvement):
    process_method(dataset_id, model, improvement)


def main(dataset_id, wanted_min_relative_improvement, memory_limit_mb, time_limit_seconds):
    print("MFE - Method: Pandas, Dataset: " + str(dataset_id) + ", Model: Recursive Surrogate Model using CatBoost using Pandas")
    model = "LightGBM_BAG_L1"
    process_func = partial(run_process_method, dataset_id, model, wanted_min_relative_improvement)
    exit_code = run_with_resource_limits(process_func, mem_limit_mb=memory_limit_mb, time_limit_sec=time_limit_seconds)
    if exit_code != 0:
        print(f"[Warning] Method failed or was terminated. Skipping.\n")


def run_with_resource_limits(target_func, mem_limit_mb, time_limit_sec, check_interval=5):
    process = multiprocessing.Process(target=target_func)
    process.start()
    pid = process.pid
    while process.is_alive():
        try:
            mem = psutil.Process(pid).memory_info().rss / (1024 * 1024)  # MB
            elapsed_time = time.time() - last_reset_time.value
            if mem > mem_limit_mb:
                print(f"[Monitor] Memory exceeded: {mem:.2f} MB > {mem_limit_mb} MB. Terminating.")
                process.terminate()
                break
            if elapsed_time > time_limit_sec:
                print(f"[Monitor] Time limit exceeded: {elapsed_time:.1f} sec > {time_limit_sec} sec. Terminating.")
                process.terminate()
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(check_interval)
    process.join()
    return process.exitcode


def main_wrapper():
    dataset = 146820
    wanted_min_relative_improvement = 0.1
    memory_limit_mb = 64000
    time_limit_seconds = 1000
    main(int(dataset), wanted_min_relative_improvement, memory_limit_mb, time_limit_seconds)


if __name__ == '__main__':
    last_reset_time = Value(ctypes.c_double, time.time())
    main_wrapper()
