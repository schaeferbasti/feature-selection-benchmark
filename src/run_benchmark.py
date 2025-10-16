import glob

import pandas as pd
from collections import defaultdict

import pyarrow

from src.utils.get_data import split_data, get_openml_dataset_split_and_metadata, concat_data
from src.utils.run_models import get_model_score_origin_classification, get_model_score_origin_regression


def main():
    target_label = 'target'

    result_files = glob.glob("data/*/*.parquet")
    result_files.sort()

    dataset_files = defaultdict(list)
    for f in result_files:
        dataset_id = f.split('.')[0].split('/')[-1].split('_')[-1]
        dataset_files[dataset_id].append(f)

    for dataset_id, files in dataset_files.items():
        print(f"\nProcessing Dataset ID: {dataset_id} and files: {files}")
        X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(int(dataset_id))
        task_type = dataset_metadata["task_type"]
        print(task_type)
        data = concat_data(X_train, y_train, X_test, y_test, "target")
        data.to_parquet(f"data/original/Original_{dataset_id}.parquet")
        # === ORIGINAL RESULTS ===
        original_path = f"results/Original_{dataset_id}.parquet"
        try:
            original_results = pd.read_parquet(original_path)
        except FileNotFoundError:
            X_train_copy = X_train.copy()
            y_train_copy = y_train.copy()
            X_test_copy = X_test.copy()
            y_test_copy = y_test.copy()
            if task_type == "Supervised Classification":
                original_results = get_model_score_origin_classification(X_train_copy, y_train_copy, X_test_copy, y_test_copy, dataset_id, "Original")
            else:
                original_results = get_model_score_origin_regression(X_train, y_train, X_test, y_test, dataset_id, "Original")
            original_results = original_results[original_results['model'] == "LightGBM_BAG_L1"]
            original_results.to_parquet(f"results/Original_{dataset_id}.parquet")
        print("Original Results loaded.")

        combined_results = [original_results]

        # === METHOD RESULTS ===
        for data_file in files:
            print(f"  Processing file: {data_file}")
            name = data_file.split('.')[0]
            method_and_dataset = name.split('/')[-1]
            method_name = method_and_dataset.split('_')[0]
            result_path = f"results/{method_and_dataset}.parquet"
            try:
                results = pd.read_parquet(result_path)
            except (FileNotFoundError, pyarrow.lib.ArrowInvalid):
                try:
                    df = pd.read_parquet(data_file)
                    Xf_train, yf_train, Xf_test, yf_test = split_data(df, target_label)
                    if task_type == 'Supervised Classification':
                        results = get_model_score_origin_classification(Xf_train, yf_train, Xf_test, yf_test, dataset_id, method_name)
                    else:
                        results = get_model_score_origin_regression(Xf_train, yf_train, Xf_test, yf_test, dataset_id, method_name)
                    results = results[results['model'] == 'LightGBM_BAG_L1']
                    results.to_parquet(result_path)
                except KeyError:
                    print('No data file')
                    continue
                combined_results.append(results)
            else:
                combined_results.append(results)

        all_results = pd.concat(combined_results, ignore_index=True).drop_duplicates()
        all_results.to_parquet(f'results/Result_{dataset_id}.parquet')
        print(f'Saved combined results for dataset {dataset_id}.')


if __name__ == "__main__":
    for fold in range(1):
        main()
