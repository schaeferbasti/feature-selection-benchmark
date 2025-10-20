from src.results.analysis.analysis import analysis
from src.run_benchmark import run_benchmark
from src.run_methods import run_methods


def run_everything(dataset_ids):
    # Run all Methods
    run_methods(dataset_ids)
    # Run Benchmark
    run_benchmark()
    # Run Result Analysis
    analysis()


if __name__ == "__main__":
    dataset_ids = [146818, 146820]
    run_everything(dataset_ids)
