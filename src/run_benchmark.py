import glob
import pandas as pd
from collections import defaultdict
import pyarrow

from src.utils.get_data import split_data, get_openml_dataset_split_and_metadata, concat_data
from src.utils.run_models import get_sklearn_model_score_classification, get_sklearn_model_score_regression

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_benchmark(n_repeat, models, classification_scores, regression_scores):
    for repeat in range(n_repeat):
        print("Repeat: ", repeat)
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
            data = concat_data(X_train, y_train, X_test, y_test, "target")
            data.to_parquet(f"data/original/Original_{dataset_id}.parquet")

            original_path = f"results/Original_{dataset_id}.parquet"
            for model in models:

                # === ORIGINAL RESULTS ===
                try:
                    original_results = pd.read_parquet(original_path)
                except FileNotFoundError:
                    X_train_copy = X_train.copy()
                    y_train_copy = y_train.copy()
                    X_test_copy = X_test.copy()
                    y_test_copy = y_test.copy()
                    if task_type == "Supervised Classification":
                        original_results = get_sklearn_model_score_classification(X_train_copy, y_train_copy, X_test_copy, y_test_copy, dataset_id, "Original", model, repeat, classification_scores)
                    else:
                        original_results = get_sklearn_model_score_regression(X_train, y_train, X_test, y_test, dataset_id, "Original", model, repeat, regression_scores)
                    original_results = original_results[original_results['model'] == "LightGBM_BAG_L1"]
                    original_results.to_parquet(original_path)
                combined_results = [original_results]

                # === METHOD RESULTS ===
                for data_file in files:
                    print(f"  Processing file: {data_file}")
                    name = data_file.split('.')[0]
                    method_and_dataset = name.split('/')[-1]
                    method_name = method_and_dataset.split('_')[0]
                    result_path = f"results/{method_and_dataset}.parquet"

                    try:
                        existing_results = pd.read_parquet(result_path)
                    except (FileNotFoundError, pyarrow.lib.ArrowInvalid):
                        existing_results = None

                    scores_to_check = classification_scores if task_type == 'Supervised Classification' else regression_scores
                    if existing_results is not None:
                        mask_fold_model = (existing_results["seed"] == repeat) & (existing_results["model"] == model)
                        if mask_fold_model.any():
                            # Fold+model exists: check for missing scores
                            existing_scores = set(existing_results.loc[mask_fold_model, "score_name"].unique())
                            missing_scores = [s for s in scores_to_check if s not in existing_scores]
                        else:
                            # Fold+model doesn't exist: need all scores
                            missing_scores = list(scores_to_check)
                    else:
                        # No file exists: need all scores
                        missing_scores = list(scores_to_check)

                    # Compute results only if there are missing scores or no data file exists
                    if missing_scores:  # or existing_results is None:
                        try:
                            df = pd.read_parquet(data_file)
                            Xf_train, yf_train, Xf_test, yf_test = split_data(df, target_label, repeat)
                            if task_type == 'Supervised Classification':
                                results = get_sklearn_model_score_classification(Xf_train, yf_train, Xf_test, yf_test, dataset_id, method_name, model, repeat, missing_scores)
                            else:
                                results = get_sklearn_model_score_regression(Xf_train, yf_train, Xf_test, yf_test, dataset_id, method_name, model, repeat, missing_scores)

                            # If existing results: concat and save; otherwise just save
                            if existing_results is not None:
                                existing_results = pd.concat([existing_results, results], ignore_index=True)
                                existing_results = existing_results.drop_duplicates(
                                    subset=['seed', 'model', 'score_name'], keep='first')
                            else:
                                existing_results = results
                            existing_results.to_parquet(result_path)
                        except KeyError as e:
                            print(f'No data file or KeyError: {e}')
                            continue
                    else:
                        print(f"All scores present for {method_name}, fold {repeat}, model {model}")
                    # Append results to combined_results (only once)
                    if existing_results is not None:
                        combined_results.append(existing_results)

                # Save final combined results
                all_results = pd.concat(combined_results, ignore_index=True).drop_duplicates()
                all_results.to_parquet(f'results/Result_{dataset_id}.parquet')


if __name__ == "__main__":
    folds = 2
    models = ["HistGradientBoosting", "RandomForest"]
    classification_scores = ["log_loss", "roc_auc_score"]
    regression_scores = ["root_mean_squared_error", "max_error"]
    run_benchmark(folds, models, classification_scores, regression_scores)
