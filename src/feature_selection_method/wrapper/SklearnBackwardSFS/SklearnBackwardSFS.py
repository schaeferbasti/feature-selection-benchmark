import multiprocessing
import time
from functools import partial

import psutil
from multiprocessing import Value
import ctypes

import pandas as pd

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

from src.utils.get_data import concat_data, get_openml_dataset_split_and_metadata

last_reset_time = Value(ctypes.c_double, time.time())


def process_method(dataset_id):
    last_reset_time.value = time.time()
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
    # Backward Sequential Feature Selector
    print("Wrapper Method: Backward Sequential Feature Selector, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("data/wrapper/SklearnBackwardSFS_" + str(dataset_id) + ".parquet")  # ../
        print("File exists" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection")
        sfs = SequentialFeatureSelector(
            estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train),
            direction='backward', cv=10)
        sfs = sfs.fit(X_train, y_train)
        selected_features = X_train.columns[sfs.get_support()]
        X_train_new = X_train[selected_features]
        X_test_new = X_test[selected_features]
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("data/wrapper/SklearnBackwardSFS_" + str(dataset_id) + ".parquet")  # ../
        print("File created" + str(data.head()) + "\n\n")

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


def main(dataset_id):
    memory_limit_mb = 64000
    time_limit_seconds = 1000
    process_func = partial(process_method, dataset_id)
    exit_code = run_with_resource_limits(process_func, mem_limit_mb=memory_limit_mb, time_limit_sec=time_limit_seconds)
    if exit_code != 0:
        print(f"[Warning] Method failed or was terminated. Skipping.\n")


if __name__ == '__main__':
    last_reset_time = Value(ctypes.c_double, time.time())
    dataset_id = 146820
    main(dataset_id)
