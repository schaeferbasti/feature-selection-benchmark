from src.results.analysis.analysis import analysis
from src.run_benchmark import run_benchmark
from src.run_methods import run_methods


def run_everything(dataset_ids, models, classification_scores, regression_scores):
    # Run all Methods
    run_methods(dataset_ids)
    # Run Benchmark
    run_benchmark(models, classification_scores, regression_scores)
    # Run Result Analysis
    analysis()


if __name__ == "__main__":
    dataset_ids = [2, 146818, 146820, 167120, 167210, 168350, 168757, 168784, 189354]  # 2073, 190146, 233211, 359930, 359931, 359932, 359933, 359935, 359936, 359937, 359938, 359944, 359949, 359950, 359952, 359954, 359955, 359956, 359958, 359959, 359960, 359962, 359963, 359965, 359968, 359971, 359972, 359974, 359975, 359979, 359981, 359982, 359983, 359987, 359992, 359993]
    # models = ["LightGBM_BAG_L1"]
    models = ["LightGBM", "RandomForest"]  # , "MLP", "SVM", "Naive Bayes", "KNN"]
    classification_scores = ["log_loss"]
    regression_scores = ["root_mean_squared_error"]
    run_everything(dataset_ids, models, classification_scores, regression_scores)
