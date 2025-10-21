import glob
import importlib
import time

import pandas as pd

def run_methods(dataset_ids):
    methods = glob.glob("feature_selection_method/*/*/*.py")
    with open("results/times.txt", 'a') as f:
        f.write(f" *************************************************** \n Time per Method and Dataset \n *************************************************** \n\n")
    for method in methods:
        if any(skip in method for skip in ["Add_Pandas_Metafeatures.py", "MUFS.py", "Metrics.py", "MAFESE.py"]):
            continue
        method_name = method.split("/")[-1].split(".")[0]
        path_name = method.split('method/')[-1].replace(str("/" + method_name + ".py"), '')
        print("Method: " + method_name)
        module_path = f"src.feature_selection_method.{path_name.replace('/', '.')}.{method_name}"

        for dataset_id in dataset_ids:
            output_path = f"data/{path_name}_{dataset_id}.parquet"
            try:
                pd.read_parquet(output_path)
                print(f"✅ File exists for {method_name}, dataset {dataset_id}")
            except FileNotFoundError:
                print(f"⚠️ File missing for dataset {dataset_id} — generating with {method_name}.main()")
                module = importlib.import_module(module_path)
                start = time.time()
                module.main(dataset_id)
                end = time.time()
                method_time = end - start
                with open("results/times.txt", 'a') as f:
                    f.write(f"{method_name} - {dataset_id}: {method_time}\n")


if __name__ == "__main__":
    dataset_ids = [146818]
    run_methods(dataset_ids)
