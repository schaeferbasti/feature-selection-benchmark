import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

from src.utils.get_data import concat_data, get_dataset_split, get_openml_dataset_split_and_metadata
from src.utils.run_models import get_sklearn_model_score_classification, get_sklearn_model_score_regression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

LOWER_BETTER = ["log_loss", "root_mean_squared_error", "max_error"]
HIGHER_BETTER = ["roc_auc_score"]

def get_metric_direction(score_name):
    if score_name in LOWER_BETTER:
        return "lower"
    elif score_name in HIGHER_BETTER:
        return "higher"
    else:
        # Default to lower is better
        return "lower"


def evaluate_subset(X_train, y_train, X_test, y_test, dataset_id, task_type, model_name, score_name, fold):
    if task_type == "Supervised Classification":
        results = get_sklearn_model_score_classification(X_train, y_train, X_test, y_test, dataset_id, "EnumerateFS", model_name, fold, score_name)
    else:
        results = get_sklearn_model_score_regression(X_train, y_train, X_test, y_test, dataset_id, "EnumerateFS", model_name, fold, score_name)
    return results["score_test"].iloc[0]


def enumerateFS(X_train, y_train, X_test, y_test, dataset_id, task_type, model_name, score_name, fold):
    n_features = X_train.shape[1]
    if n_features <= 10:
        direction = get_metric_direction(score_name)
        if direction == "lower":
            best_score = np.inf
        if direction == "higher":
            best_score = -np.inf
        best_features = None
        current_combo = 0
        feature_names = X_train.columns.tolist()

        # Enumerate all possible subsets (from 1 to max_features)
        for selected_n_features in range(1, min(n_features + 1, n_features + 1)):
            for feature_combo in combinations(range(n_features), selected_n_features):
                feature_indices = list(feature_combo)
                print(feature_indices)
                X_train_selection = X_train.iloc[:, feature_indices]
                X_test_selection = X_test.iloc[:, feature_indices]
                score = evaluate_subset(X_train_selection, y_train, X_test_selection, y_test, dataset_id, task_type, model_name, score_name, fold)
                selected_feature_names = [feature_names[i] for i in feature_indices]
                if direction == "lower":
                    if score < best_score:
                        best_score = score
                        best_indices = feature_indices
                        best_features = selected_feature_names
                if direction == "higher":
                    if score > best_score:
                        best_score = score
                        best_indices = feature_indices
                        best_features = selected_feature_names
                current_combo += 1
        print(str(best_score))
        return best_features
    else:
        return None


def main(dataset_id):
    model_names = ["HistGradientBoosting", "RandomForest"]
    classification_scores = ["log_loss", "roc_auc_score"]
    regression_scores = ["root_mean_squared_error", "max_error"]
    seeds = 1
    for seed in range(seeds):
        X_train, y_train, X_test, y_test, dataset_metadata = get_dataset_split(dataset_id, seed)
        task_type = dataset_metadata["task_type"]
        if task_type == "Supervised Classification":
            for score_name in classification_scores:
                iterate_models(X_test, X_train, dataset_id, model_names, score_name, seeds, task_type, y_test, y_train)
        else:
            for score_name in regression_scores:
                iterate_models(X_test, X_train, dataset_id, model_names, score_name, seeds, task_type, y_test, y_train)


def iterate_models(X_test, X_train, dataset_id, model_names, score_name, seed, task_type, y_test, y_train):
    for model_name in model_names:
        print(f"Enumeration Method, Model: {model_name}, Score: {score_name}, Seed: {seed}, Dataset: {dataset_id}")
        try:
            data = pd.read_parquet(f"data/enumeration/enumFS_{dataset_id}_{model_name}_{score_name}_{seed}.parquet")  # ../../../
            print("File exists" + str(data.head()) + "\n\n")
        except FileNotFoundError:
            print("Calculate Feature Selection")
            selected_features = enumerateFS(X_train, y_train, X_test, y_test, dataset_id, task_type, model_name, score_name, seed)
            if selected_features is None:
                pass
            else:
                X_train_new = pd.DataFrame(X_train, columns=selected_features, index=X_train.index)
                X_test_new = pd.DataFrame(X_test, columns=selected_features, index=X_test.index)
                data = concat_data(X_train_new, y_train, X_test_new, y_test, "target")
                data.to_parquet(f"data/enumeration/enumFS_{dataset_id}_{model_name}_{score_name}_{seed}.parquet")  # ../../../
                print("File created" + str(data.head()) + "\n\n")


if __name__ == '__main__':
    dataset_id = 146820
    main(dataset_id)
