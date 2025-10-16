from __future__ import annotations

import multiprocessing
import time
from functools import partial

import pandas as pd
import psutil

from Add_Pandas_Metafeatures import add_pandas_metadata_selection_columns
from autogluon.tabular.models import CatBoostModel

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


def remove_feature(X_train, y_train, X_test, y_test, prediction_result):
    # Get feature with the lowest predicted improvement
    worst_feature_row = prediction_result.loc[prediction_result['predicted_improvement'].idxmin()]
    worst_feature_name = worst_feature_row['feature - name'].split(' - ')[-1]
    print(f"Removing worst feature: {worst_feature_name} (predicted_improvement={worst_feature_row['predicted_improvement']:.4f})")
    # Drop the column if it exists in X
    if worst_feature_name in X_train.columns:
        X_train = X_train.drop(columns=[worst_feature_name])
        X_test = X_test.drop(columns=[worst_feature_name])
    else:
        print(f"Warning: Feature '{worst_feature_name}' not found in X.")
    return X_train, y_train, X_test, y_test


def execute_feature_selection_recursive(prediction_result, X_train, y_train, X_test, y_test,):
    dataset_id = int(prediction_result["dataset - id"].values[0])
    model = prediction_result["model"].values[0]
    X_train, y_train, X_test, y_test = remove_feature(X_train, y_train, X_test, y_test, prediction_result)
    return X_train, y_train, X_test, y_test, dataset_id, model


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
    return comparison_result_matrix


def recursive_feature_selection(X_train, y_train, X_test, y_test, model, dataset_metadata, category_to_drop, dataset_id):
    result_matrix = pd.read_parquet("Pandas_Matrix_Complete.parquet")
    datasets = pd.unique(result_matrix["dataset - id"]).tolist()
    if dataset_id in datasets:
        result_matrix = result_matrix[result_matrix["dataset - id"] != dataset_id]
    comparison_result_matrix = create_empty_core_matrix_for_dataset(X_train, model, dataset_id)
    comparison_result_matrix = add_pandas_metadata_selection_columns(dataset_metadata, X_train, comparison_result_matrix)
    # Predict and split again
    X_train_new, y_train_new, X_test_new, y_test_new = predict_improvement(result_matrix, comparison_result_matrix, X_train, y_train, X_test, y_test)
    if X_train_new.equals(X_train):
        try:
            y_list = y_train['target'].tolist()
            y_series = pd.Series(y_list)
            y_train = y_series
        except KeyError:
            print("")
        data = concat_data(X_train, y_train, X_test, y_test, "target")
        data.to_parquet("../../data/metalearning/MetaFS_" + str(dataset_id) + ".parquet")
        print("Write File: data/metalearning/MetaFS_" + str(dataset_id) + ".parquet")
        return X_train_new, y_train_new, X_test_new, y_test_new
    else:
        try:
            y_list = y_train_new['target'].tolist()
            y_series = pd.Series(y_list)
            y_train_new = y_series
        except KeyError:
            print("")
        print(X_train_new.columns)
        data = concat_data(X_train_new, y_train_new, X_test_new, y_test_new, "target")
        data.to_parquet("../../data/metalearning/MetaFS_" + str(dataset_id) + ".parquet")
        print("Write File: data/metalearning/MetaFS_" + str(dataset_id) + ".parquet")
        return recursive_feature_selection(X_train_new, y_train_new, X_test_new, y_test_new, model, dataset_metadata, category_to_drop, dataset_id)


def predict_improvement(result_matrix, comparison_result_matrix, X_train, y_train, X_test, y_test):
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
    if best_operation["predicted_improvement"].values[0] < 0:
        print("Predicted improvement of least important feature: " + str(
            best_operation["predicted_improvement"].values[0]) + " - not good enough")
        return X_train, y_train, X_test, y_test
    else:
        print("Predicted improvement of least important feature: " + str(
            best_operation["predicted_improvement"].values[0]) + " - execute feature engineering")
        X_train, y_train, X_test, y_test, _, _ = execute_feature_selection_recursive(best_operation, X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test


def run_process_method(dataset_id, model):
    try:
        pd.read_parquet("../../data/metalearning/MetaFS_" + str(dataset_id) + ".parquet")
    except FileNotFoundError:
        last_reset_time.value = time.time()
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
        X_train, y_train, X_test, y_test = recursive_feature_selection(X_train, y_train, X_test, y_test, model, dataset_metadata, None, dataset_id)
        data = concat_data(X_train, y_train, X_test, y_test, "target")
        data.to_parquet("../../data/metalearning/MetaFS_" + str(dataset_id) + ".parquet")
        print("Write File: data/metalearning/MetaFS_" + str(dataset_id) + ".parquet")


def main(dataset_id, memory_limit_mb, time_limit_seconds):
    print("MFE - Method: Pandas, Dataset: " + str(dataset_id) + ", Model: Recursive Surrogate Model using CatBoost using Pandas")
    model = "LightGBM_BAG_L1"
    process_func = partial(run_process_method, dataset_id, model)
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
    wanted_max_relative_downgrade = 0.1
    memory_limit_mb = 64000
    time_limit_seconds = 1000
    main(int(dataset), memory_limit_mb, time_limit_seconds)


if __name__ == '__main__':
    last_reset_time = Value(ctypes.c_double, time.time())
    main_wrapper()
