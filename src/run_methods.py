import glob
import importlib
import time

import pandas as pd

import subprocess


def send_notification(title, message):
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(['osascript', '-e', script])

def run_methods(dataset_ids):
    methods = glob.glob("feature_selection_method/*/*/*.py")
    for method in methods:
        if any(skip in method for skip in ["Add_Pandas_Metafeatures.py", "MUFS.py", "Metrics.py", "MAFESE.py", "BioAutoML.py", "MACFE.py"]):
            continue
        method_name = method.split("/")[-1].split(".")[0]
        path_name = method.split('method/')[-1].replace(str("/" + method_name + ".py"), '')
        print("Method: " + method_name)
        module_path = f"src.feature_selection_method.{path_name.replace('/', '.')}.{method_name}"

        for dataset_id in dataset_ids:
            if method_name == "SklearnBackwardSFS" and dataset_id == 167210:
                continue
            elif method_name == "SklearnBackwardSFS" and dataset_id == 168757:
                continue
            elif method_name == "SklearnBackwardSFS" and dataset_id == 189354:
                continue
            elif method_name == "SklearnExtraTreesClassifier" and dataset_id == 167210:
                continue
            elif method_name == "SklearnExtraTreesClassifier" and dataset_id == 168757:
                continue
            elif method_name == "SklearnExtraTreesClassifier" and dataset_id == 189354:
                continue
            elif method_name == "SklearnForwardSFS" and dataset_id == 167210:
                continue
            elif method_name == "SklearnForwardSFS" and dataset_id == 168757:
                continue
            elif method_name == "SklearnForwardSFS" and dataset_id == 189354:
                continue
            elif method_name == "MACFE" and dataset_id == 146818:
                continue
            elif method_name == "SklearnLinearSVC" and dataset_id == 167210:
                continue
            elif method_name == "SklearnLinearSVC" and dataset_id == 168757:
                continue
            elif method_name == "SklearnLinearSVC" and dataset_id == 189354:
                continue
            elif method_name == "MetaFE" and dataset_id == 167120:
                continue
            elif method_name == "MetaFE" and dataset_id == 167210:
                continue
            elif method_name == "MetaFE" and dataset_id == 168757:
                continue
            elif method_name == "MetaFE" and dataset_id == 189354:
                continue
            elif method_name == "mRMR" and dataset_id == 167210:
                continue
            elif method_name == "mRMR" and dataset_id == 168757:
                continue
            elif method_name == "mRMR" and dataset_id == 189354:
                continue
            elif method_name == "CorrelationBasedFS" and dataset_id == 189354:
                continue
            elif method_name == "SkrebateReliefF":
                continue
            elif method_name == "SklearnSelectPercentileMutualInfoClassif":
                continue
            elif method_name == "SklearnSelectKBestClassifF":
                continue
            elif method_name == "SkrebateSURF":
                continue
            elif method_name == "SklearnSelectKBestChi2":
                continue
            elif method_name == "SklearnVarianceThreshold" and dataset_id == 167120:
                continue
            elif method_name == "SklearnVarianceThreshold" and dataset_id == 167210:
                continue
            elif method_name == "SklearnVarianceThreshold" and dataset_id == 168757:
                continue
            elif method_name == "SklearnVarianceThreshold" and dataset_id == 189354:
                continue
            elif method_name == "SklearnRFE":
                continue
            elif method_name == "SklearnRFECV":
                continue
            else:
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
                        send_notification("File saved", f"{method_name} - {dataset_id}: {method_time}")


if __name__ == "__main__":
    dataset_ids = [146818]
    run_methods(dataset_ids)
