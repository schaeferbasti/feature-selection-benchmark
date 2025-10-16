import multiprocessing
import time
from functools import partial

import psutil
from multiprocessing import Value
import ctypes

import pandas as pd

from sklearn.feature_selection import SelectKBest, SelectFromModel, VarianceThreshold, SelectPercentile, RFECV, RFE
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

from src.utils.get_data import concat_data, get_openml_dataset_split_and_metadata

last_reset_time = Value(ctypes.c_double, time.time())

def process_method(dataset_id):
    last_reset_time.value = time.time()
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)

    # Variance Threshold
    print("Filter Method: Variance Threshold, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("../data/filter/SklearnVarianceThreshold_" + str(dataset_id) + ".parquet")
        print("File exists, next method" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection \n\n")
        variance_threshold = VarianceThreshold(threshold=(.8 * (1 - .8)))
        X_train_new = variance_threshold.fit_transform(X_train)
        X_test_new = variance_threshold.transform(X_test)
        selected_features = X_train.columns[variance_threshold.get_support()]
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("../data/filter/SklearnVarianceThreshold_" + str(dataset_id) + ".parquet")

    # SelectKBest - Classif_f
    print("Filter Method: SelectKBest, Score Function: Classif F, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("../data/filter/SklearnSelectKBestFClassif_" + str(dataset_id) + ".parquet")
        print("File exists, next method" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection \n\n")
        selectKBest = SelectKBest(score_func=f_classif, k=3)
        selectKBest.fit(X_train, y_train)
        X_train_new = selectKBest.transform(X_train)
        X_test_new = selectKBest.transform(X_test)
        selected_features = X_train.columns[selectKBest.get_support()]
        # Transform the data and wrap it back into DataFrames
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("../data/filter/SklearnSelectKBestFClassif_" + str(dataset_id) + ".parquet")

    # SelectKBest - Chi2
    print("Filter Method: SelectKBest, Score Function: Chi2, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("../data/filter/SklearnSelectKBestChi2_" + str(dataset_id) + ".parquet")
        print("File exists, next method" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection \n\n")
        selectKBest = SelectKBest(score_func=chi2, k=3)
        selectKBest.fit(X_train, y_train)
        X_train_new = selectKBest.transform(X_train)
        X_test_new = selectKBest.transform(X_test)
        selected_features = X_train.columns[selectKBest.get_support()]
        # Transform the data and wrap it back into DataFrames
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("../data/filter/SklearnSelectKBestChi2_" + str(dataset_id) + ".parquet")

    # SelectPercentile - Mutual_info_classif
    print("Filter Method: SelectPercentile, Score Function: Mutual Info Classifier, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("../data/filter/SklearnSelectPercentileMutualInfo_" + str(dataset_id) + ".parquet")
        print("File exists, next method" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection \n\n")
        selectKBest = SelectPercentile(score_func=mutual_info_classif, percentile=50)
        selectKBest.fit(X_train, y_train)
        X_train_new = selectKBest.transform(X_train)
        X_test_new = selectKBest.transform(X_test)
        selected_features = X_train.columns[selectKBest.get_support()]
        # Transform the data and wrap it back into DataFrames
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("../data/filter/SklearnSelectPercentileMutualInfo_" + str(dataset_id) + ".parquet")

    # RFECV
    print("Filter Method: Recursive Feature Elimination, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("../data/embedded/SklearnRFE_" + str(dataset_id) + ".parquet")
        print("File exists, next method" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection \n\n")
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), step=1)
        rfe.fit(X_train, y_train)
        X_train_new = rfe.transform(X_train)
        X_test_new = rfe.transform(X_test)
        selected_features = X_train.columns[rfe.get_support()]
        # Transform the data and wrap it back into DataFrames
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("../data/embedded/SklearnRFE_" + str(dataset_id) + ".parquet")

    # RFECV
    print("Filter Method: Recursive Feature Elimination with CV, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("../data/embedded/SklearnRFECV_" + str(dataset_id) + ".parquet")
        print("File exists, next method" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection \n\n")
        rfecv = RFECV(estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), step=1, min_features_to_select=1)
        rfecv.fit(X_train, y_train)
        X_train_new = rfecv.transform(X_train)
        X_test_new = rfecv.transform(X_test)
        selected_features = X_train.columns[rfecv.get_support()]
        # Transform the data and wrap it back into DataFrames
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("../data/embedded/SklearnRFECV_" + str(dataset_id) + ".parquet")

    # Select From Model (Linear SVC)
    print("Filter Method: Linear SVC, Penalty: L1, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("../data/wrapper/SklearnLinearSVC_" + str(dataset_id) + ".parquet")
        print("File exists, next method" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection \n\n")
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
        model = SelectFromModel(lsvc, prefit=True)
        X_train_new = model.transform(X_train)
        X_test_new = model.transform(X_test)
        selected_features = X_train.columns[model.get_support()]
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("../data/wrapper/SklearnLinearSVC_" + str(dataset_id) + ".parquet")

    # Select From Model (Extra Trees Classifier)
    print("Filter Method: ExtraTreeClassifier, Dataset: " + str(dataset_id))
    try:
        data = pd.read_parquet("../data/wrapper/SklearnExtraTreeClassifier_" + str(dataset_id) + ".parquet")
        print("File exists, next method" + str(data.head()) + "\n\n")
    except FileNotFoundError:
        print("Calculate Feature Selection \n\n")
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X_train, y_train)
        model = SelectFromModel(clf, prefit=True)
        X_train_new = model.transform(X_train)
        X_test_new = model.transform(X_test)
        selected_features = X_train.columns[model.get_support()]
        X_train_new = pd.DataFrame(X_train_new, columns=selected_features, index=X_train.index)
        X_test_new = pd.DataFrame(X_test_new, columns=selected_features, index=X_test.index)
        data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
        data.to_parquet("../data/wrapper/SklearnExtraTreeClassifier_" + str(dataset_id) + ".parquet")


def run_process_method(dataset_id):
    process_method(dataset_id)


def main(dataset_id, memory_limit_mb, time_limit_seconds):
    process_func = partial(run_process_method, dataset_id)
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
    memory_limit_mb = 64000
    time_limit_seconds = 1000
    main(int(dataset), memory_limit_mb, time_limit_seconds)


if __name__ == '__main__':
    last_reset_time = Value(ctypes.c_double, time.time())
    main_wrapper()
