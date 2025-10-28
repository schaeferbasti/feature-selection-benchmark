import glob

import matplotlib
import numpy as np
import openml
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.cm as cm
from numpy import ndarray


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


def get_data(result_files):
    all_results = []
    for result_file in result_files:
        df = pd.read_parquet(result_file)
        dataset_id = int(result_file.split("results/Result_")[1].split(".parquet")[0])
        all_results.append(df)
    df_all = pd.concat(all_results, ignore_index=True)

    # Convert score to error
    df_all["error_val"] = - df_all["score_val"]
    df_all["error_test"] = - df_all["score_test"]

    # Remove duplicates
    df_all = df_all.drop_duplicates()

    # Group by dataset, model, score_name, and origin
    # Compute mean and std across folds (seed)
    df_grouped = df_all.groupby(["dataset", "model", "score_name", "origin"]).agg({
        "score_val": ["mean", "std"],
        "score_test": ["mean", "std"]
    }).reset_index()

    # Flatten column names
    df_grouped.columns = ['_'.join(col).strip('_') for col in df_grouped.columns.values]

    return df_grouped


def separate_by_score_and_pivot(df_grouped):
    """
    Separate by score_name, then pivot each score separately.
    Returns dict: {score_name: (df_pivot_val, df_pivot_val_std, df_pivot_test, df_pivot_test_std, dataset_list)}
    """
    result = {}
    score_names = df_grouped['score_name'].unique()

    for score in score_names:
        # Filter for this score only
        df_score = df_grouped[df_grouped['score_name'] == score].copy()

        # Create instance identifier (dataset|model)
        df_score["instance"] = df_score.apply(
            lambda row: f"{row['dataset']}|{row['model']}",
            axis=1
        )

        # Pivot for validation scores (mean)
        df_pivot_val = df_score.pivot(index="instance", columns="origin", values="score_val_mean")
        df_pivot_val = df_pivot_val.sort_index()
        df_pivot_val = make_model_name_nice(df_pivot_val)

        # Pivot for validation scores (std)
        df_pivot_val_std = df_score.pivot(index="instance", columns="origin", values="score_val_std")
        df_pivot_val_std = df_pivot_val_std.sort_index()
        df_pivot_val_std = make_model_name_nice(df_pivot_val_std)

        # Pivot for test scores (mean)
        df_pivot_test = df_score.pivot(index="instance", columns="origin", values="score_test_mean")
        df_pivot_test = df_pivot_test.sort_index()
        df_pivot_test = make_model_name_nice(df_pivot_test)

        # Pivot for test scores (std)
        df_pivot_test_std = df_score.pivot(index="instance", columns="origin", values="score_test_std")
        df_pivot_test_std = df_pivot_test_std.sort_index()
        df_pivot_test_std = make_model_name_nice(df_pivot_test_std)

        # Get dataset names for display
        datasets = df_pivot_val.index.astype(str)
        dataset_list = []
        for dataset in datasets.tolist():
            parts = dataset.split('|')
            dataset_id = int(parts[0])
            model_name = parts[1]

            try:
                task = openml.tasks.get_task(
                    dataset_id,
                    download_splits=True,
                    download_data=True,
                    download_qualities=True,
                    download_features_meta_data=True,
                )
                dataset_name = task.get_dataset().name
                # Create readable label: dataset_name | model
                label = f"{dataset_name}|{model_name}"
                dataset_list.append(label)
            except Exception as e:
                print(f"Error loading dataset {dataset_id}: {e}")
                dataset_list.append(dataset)

        result[score] = (df_pivot_val, df_pivot_val_std, df_pivot_test, df_pivot_test_std, dataset_list)

    return result


def make_model_name_nice(df_pivot):
    model_names_nice = []
    model_names = df_pivot.columns
    for model_name in model_names:
        model_name = model_name.replace('pandas_', 'Pandas, ')
        model_name = model_name.replace('d2v_', 'Dataset2Vec, ')
        model_name = model_name.replace('tabpfn_', 'TabPFN, ')
        model_name = model_name.replace('MFE_general_', 'MFE (general), ')
        model_name = model_name.replace('MFE_statistical', 'MFE (statistical), ')
        model_name = model_name.replace('MFE_info-theory', 'MFE (info-theory), ')
        model_name = model_name.replace("MFE_{'general', 'info-theory'}", 'MFE (general, info-theory), ')
        model_name = model_name.replace("MFE_{'statistical', 'info-theory'}", 'MFE (statistical, info-theory), ')
        model_name = model_name.replace("MFE_{'info-theory', 'general'}", 'MFE (info-theory, general), ')
        model_name = model_name.replace("MFE_{'info-theory', 'statistical'}", 'MFE (info-theory, statistical), ')
        model_name = model_name.replace("MFE_{'general', 'statistical'}", 'MFE (general, statistical), ')
        model_name = model_name.replace("MFE_{'statistical', 'general'}", 'MFE (statistical, general), ')
        model_name = model_name.replace("MFE_{'general', 'statistical', 'info-theory'}",
                                        'MFE (general, statistical, info-theory), ')
        model_name = model_name.replace('best', 'one-shot SM')
        model_name = model_name.replace('_one-shot', 'one-shot')
        model_name = model_name.replace('recursion', 'recursive SM')
        model_name = model_name.replace('_recursive', 'recursive')
        model_names_nice.append(model_name)
    df_pivot.columns = model_names_nice
    return df_pivot


def insert_line_breaks(name, max_len=20):
    if len(name) > max_len:
        # Split into chunks of `max_len`, preserving words if possible
        parts = [name[i:i + max_len] for i in range(0, len(name), max_len)]
        return '\n'.join(parts)
    else:
        return name


def make_latex_table(df_pivot, without_openfe):
    formatted_df = df_pivot.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "/")
    latex_lines = []
    latex_lines.append(r"\begin{table}[h!]")
    latex_lines.append(r"    \tiny")
    if without_openfe:
        latex_lines.append(
            r"        \begin{tabular*}{\textwidth}{@{\extracolsep{0.4em}} c|cccccccccccccccccccccccc @{}}")
        latex_lines.append(r"        \toprule")
        latex_lines.append(
            r"        Dataset & \makecell{Best\\Random} & \makecell{MFE\\(general),\\one-shot SM} & \makecell{MFE\\(general),\\recursive SM} & \makecell{MFE\\(info-theory),\\one-shot SM} & \makecell{MFE\\(info-theory),\\recursive SM}  & \makecell{MFE\\(statistical),\\one-shot SM}  & \makecell{MFE\\(statistical),\\recursive SM}  & \makecell{MFE\\(general, info-theory),\\one-shot SM}  & \makecell{MFE\\(general, info-theory),\\recursive SM}  & \makecell{MFE\\(general, statistical),\\one-shot SM}  & \makecell{MFE\\(general, statistical),\\recursive SM}  & \makecell{MFE\\(info-theory, general),\\one-shot SM}  & \makecell{MFE\\(info-theory, general),\\recursive SM}  & \makecell{MFE\\(info-theory, statistical),\\one-shot SM}  & \makecell{MFE\\(info-theory, statistical),\\recursive SM}  & \makecell{MFE\\(statistical, general),\\one-shot SM}  & \makecell{MFE\\(statistical, general),\\recursive SM} & \makecell{MFE\\(statistical,info-theory),\\one-shot SM} & \makecell{MFE\\(statistical, info-theory),\\recursive SM} & \makecell{Original} & \makecell{Dataset2Vec,\\one-shot SM} & \makecell{Dataset2Vec,\\recursive SM} & \makecell{Pandas,\\one-shot SM} & \makecell{Pandas,\\recursive SM} \\")

    else:
        latex_lines.append(
            r"        \begin{tabular*}{\textwidth}{@{\extracolsep{0.4em}} c|ccccccccccccccccccccccccc @{}}")
        latex_lines.append(r"        \toprule")
        latex_lines.append(
            r"        Dataset & \makecell{Best\\Random} & \makecell{MFE\\(general),\\one-shot SM} & \makecell{MFE\\(general),\\recursive SM} & \makecell{MFE\\(info-theory),\\one-shot SM} & \makecell{MFE\\(info-theory),\\recursive SM}  & \makecell{MFE\\(statistical),\\one-shot SM}  & \makecell{MFE\\(statistical),\\recursive SM}  & \makecell{MFE\\(general, info-theory),\\one-shot SM}  & \makecell{MFE\\(general, info-theory),\\recursive SM}  & \makecell{MFE\\(general, statistical),\\one-shot SM}  & \makecell{MFE\\(general, statistical),\\recursive SM}  & \makecell{MFE\\(info-theory, general),\\one-shot SM}  & \makecell{MFE\\(info-theory, general),\\recursive SM}  & \makecell{MFE\\(info-theory, statistical),\\one-shot SM}  & \makecell{MFE\\(info-theory, statistical),\\recursive SM}  & \makecell{MFE\\(statistical, general),\\one-shot SM}  & \makecell{MFE\\(statistical, general),\\recursive SM} & \makecell{MFE\\(statistical,info-theory),\\one-shot SM} & \makecell{MFE\\(statistical, info-theory),\\recursive SM} & \makecell{Original} & \makecell{Dataset2Vec,\\one-shot SM} & \makecell{Dataset2Vec,\\recursive SM} & \makecell{Pandas,\\one-shot SM} & \makecell{Pandas,\\recursive SM} & \makecell{OpenFE} \\")

    latex_lines.append(r"        \midrule")

    # Add table rows
    for dataset_id, row in formatted_df.iterrows():
        row_str = f"        {dataset_id} & " + " & ".join(row.values) + r" \\ \midrule"
        latex_lines.append(row_str)

    # Finish LaTeX code
    latex_lines.append(r"    \end{tabular*}")
    if without_openfe:
        latex_lines.append(
            r"    \caption{Test error of the model on the feature-engineered datasets of the \sm{} approaches using \metafeatures{} of the tested extractors, on the best randomly feature-engineered datasets and on the original datasets}")
        latex_lines.append(r"    \label{tab:test_without_openfe}")
    else:
        latex_lines.append(
            r"    \caption{Test error of the model on the feature-engineered datasets of the \sm{} approaches using \metafeatures{} of the tested extractors, on the best randomly feature-engineered datasets, on the original datasets, and on the datasets feature-engineered with \gls{OpenFE}}")
        latex_lines.append(r"    \label{tab:test}")
    latex_lines.append(r"\end{table}")

    latex_code = "\n".join(latex_lines)

    print(latex_code)


def make_latex_tables_split(df_pivot, without_openfe, columns_per_table=6):
    formatted_df = df_pivot.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "/")
    method_columns = df_pivot.columns.tolist()
    total_tables = 4

    for table_idx in range(total_tables):
        start_col = table_idx * columns_per_table
        # Fix: Add all remaining columns to the last table
        if table_idx == total_tables - 1:
            end_col = len(method_columns)
        else:
            end_col = start_col + columns_per_table

        current_columns = method_columns[start_col:end_col]

        latex_lines = []
        latex_lines.append(r"\begin{table}[h!]")
        latex_lines.append(r"    \footnotesize")

        column_format = "c|" + "c" * len(current_columns)
        latex_lines.append(fr"    \begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{0.2em}}}} {column_format} @{{}}}}")
        latex_lines.append(r"        \toprule")

        header_cells = ["Dataset"]
        for col in current_columns:
            escaped_col = col.replace(", ", ",\\\\").replace(" ", "\\\\")  # Optional: better breaking
            header_cells.append(f"\\makecell{{{escaped_col}}}")
        latex_lines.append("        " + " & ".join(header_cells) + r" \\")
        latex_lines.append(r"        \midrule")

        for dataset_id, row in formatted_df.iterrows():
            values = [row[col] for col in current_columns]
            row_str = f"        {dataset_id} & " + " & ".join(values) + r" \\"
            latex_lines.append(row_str)

        latex_lines.append(r"        \bottomrule")
        latex_lines.append(r"    \end{tabular*}")

        base_caption = "Test error of the model on the feature-engineered datasets"
        label_prefix = "tab:test_without_openfe" if without_openfe else "tab:test_with_openfe"
        latex_lines.append(fr"    \caption{{{base_caption} (Part {table_idx + 1})}}")
        latex_lines.append(fr"    \label{{{label_prefix}_part{table_idx + 1}}}")
        latex_lines.append(r"\end{table}")
        latex_lines.append("")

        print("\n".join(latex_lines))


"""
def make_latex_tables_as_one(df_pivot, df_pivot_std, without_openfe, columns_per_table=5):
    from math import ceil

    formatted_df = df_pivot.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "/")
    method_columns = df_pivot.columns.tolist()
    total_tables = ceil(len(method_columns) / columns_per_table)

    base_caption = "Test error of the model on the feature-engineered datasets"
    label = "tab:test_without_openfe" if without_openfe else "tab:test_with_openfe"

    for table_idx in range(total_tables):
        start_col = table_idx * columns_per_table
        end_col = min(start_col + columns_per_table, len(method_columns))
        current_columns = method_columns[start_col:end_col]

        latex_lines = []
        latex_lines.append(r"\begin{table}[h!]")
        latex_lines.append(r"    \footnotesize")

        column_format = "c|" + "c" * len(current_columns)
        latex_lines.append(fr"    \begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{0.2em}}}} {column_format} @{{}}}}")
        latex_lines.append(r"        \toprule")

        header_cells = ["Dataset"]
        for col in current_columns:
            escaped_col = col.replace(", ", ",\\\\").replace(" ", "\\\\")
            header_cells.append(f"\\makecell{{{escaped_col}}}")
        latex_lines.append("        " + " & ".join(header_cells) + r" \\")
        latex_lines.append(r"        \midrule")

        for dataset_id in df_pivot.index:
            row_cells = [dataset_id]
            for col in current_columns:
                val = df_pivot.loc[dataset_id, col]
                std = df_pivot_std.loc[dataset_id, col]
                if pd.notnull(val) and pd.notnull(std):
                    cell = f"${val:.2f} {{\\scriptscriptstyle \\pm {std:.2f}}}$"
                elif pd.notnull(val):
                    cell = f"${val:.2f}$"
                else:
                    cell = "/"
                row_cells.append(cell)
            latex_lines.append("        " + " & ".join(row_cells) + r" \\")

        latex_lines.append(r"        \bottomrule")
        latex_lines.append(r"    \end{tabular*}")

        if table_idx == 0:
            latex_lines.append(fr"    \caption{{{base_caption}}}")
            latex_lines.append(fr"    \label{{{label}}}")
        else:
            latex_lines.append(r"    \ContinuedFloat")
            latex_lines.append(fr"    \caption*{{{base_caption} (cont.)}}")

        latex_lines.append(r"\end{table}")
        latex_lines.append("")

        print("\n".join(latex_lines))
"""


def plot_score_graph(dataset_list_wrapped, df_pivot, df_pivot_std, name):
    if "without_FE" in name:
        score_type = name.split("_")[0]
        if score_type == "Val":
            score_type = "validation"
        else:
            score_type = "test"
        large_plot = True
        without_fe = True
        df_pivot = df_pivot.drop(columns=["OpenFE", "MetaFE"])
    else:
        if name == "Val":
            score_type = "validation"
        else:
            score_type = "test"
        large_plot = True
        without_fe = False
        column_to_move = df_pivot.pop("OpenFE")
        df_pivot.insert(len(df_pivot.columns), "OpenFE", column_to_move)
        # if score_type == "test":
            # make_latex_tables_as_one(df_pivot, df_pivot_std, without_openfe)
    if without_fe:
        colors = cm.get_cmap('nipy_spectral')
        color_list: list[ndarray | tuple[float, float, float, float]] = [colors(i) for i in np.linspace(0, 0.95, len(df_pivot.columns))]
    else:
        colors = cm.get_cmap('nipy_spectral', len(df_pivot.columns))

    dataset_list_wrapped = df_pivot.index.tolist()
    if large_plot:
        plt.figure(figsize=(12, 10))
        if without_fe:
            for idx, method in enumerate(df_pivot.columns):
                plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method, color=color_list[idx])
        else:
            for idx, method in enumerate(df_pivot.columns):
                plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method, color=colors(idx))
    else:
        plt.figure(figsize=(12, 7))
        for method in df_pivot.columns:
            plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method)
    plt.xlabel("Dataset")
    plt.xticks(rotation=90)  # or 45
    plt.ylabel(score_type.title() + " error")
    plt.title(
        score_type.title() + " error of the model on the feature-engineered datasets, the original and the randomly feature-engineered datasets")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/analysis/Graph_" + name + ".png")
    plt.show()


def plot_count_best(df_pivot_val, df_pivot_test, name, score_name):
    direction = get_metric_direction(score_name)
    if direction == "lower":
        minValueIndex_val = df_pivot_val.idxmin(axis=1).value_counts()
        minValueIndex_test = df_pivot_test.idxmin(axis=1).value_counts()
        title = "Count of the lowest validation/test error of the model"
    else:  # higher is better (AUC)
        minValueIndex_val = df_pivot_val.idxmax(axis=1).value_counts()
        minValueIndex_test = df_pivot_test.idxmax(axis=1).value_counts()
        title = "Count of the highest validation/test score of the model"

    plt.figure(figsize=(12, 7))
    minValueIndex_val.plot(kind='bar', color='skyblue', label=f'Validation ({direction} is better)')
    minValueIndex_test.plot(kind='bar', width=0.3, color='darkblue',
                            label=f'Test ({direction} is better)')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Number of instances")
    plt.title(title)
    plt.xticks(rotation=90, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/analysis/Count_Best_{name}bar.png")
    plt.close()


def plot_avg_percentage_impr(baseline_col, df_pivot, df_pivot_std, name, score_name, only_pandas=False):
    direction = get_metric_direction(score_name)
    score_type = "validation" if "Val" in name else "test"
    improvement = pd.DataFrame()
    for method in df_pivot.columns:
        if method == baseline_col:
            continue
        if direction == "lower":
            calc_improvement = ((df_pivot[baseline_col] - df_pivot[method]) / df_pivot[baseline_col]) * 100
        else:  # higher is better
            calc_improvement = ((df_pivot[method] - df_pivot[baseline_col]) / df_pivot[baseline_col]) * 100
        improvement[method] = calc_improvement
    avg_improvement = improvement.mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 7))
    bars = avg_improvement.plot(kind="bar", color="skyblue")
    # Labels
    for i, val in enumerate(avg_improvement):
        y = -0.1 if val >= 0 else 0
        plt.text(i, y, f"{val:.2f}%", ha='center', va='top' if val >= 0 else 'bottom', color='black', fontsize=8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(
        f"Average percentage improvement ({score_type}) – {score_name} ({direction} is better)\nrelative to baseline ({baseline_col})")
    plt.xlabel("Method")
    plt.ylabel(f"Percentage {'reduction' if direction=='lower' else 'increase'} in {score_type} {score_name}\nrelative to baseline")
    plt.xticks(rotation=90, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/analysis/Average_Percentage_Improvement_{name}.png")
    plt.close()


def plot_boxplot_percentage_impr(baseline_col, df_pivot, name, score_name):
    np.random.seed(0)
    direction = get_metric_direction(score_name)
    score_type = "validation" if "Val" in name else "test"
    improvement_test = pd.DataFrame()
    for method in df_pivot.columns:
        if method == baseline_col:
            continue
        if direction == "lower":
            improvement = (df_pivot[baseline_col] - df_pivot[method]) / df_pivot[baseline_col] * 100
        else:  # higher
            improvement = (df_pivot[method] - df_pivot[baseline_col]) / df_pivot[baseline_col] * 100
        # Clip outliers
        Q1 = improvement.quantile(0.25)
        Q3 = improvement.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        improvement_clipped = improvement.clip(lower, upper)
        improvement_test[method] = improvement
    method_order = improvement_test.mean().sort_values(ascending=False).index.tolist()
    improvement_test = improvement_test[method_order]

    plt.figure(figsize=(12, 7))
    improvement_test.boxplot(column=method_order, grid=True)
    for i, method in enumerate(method_order):
        y = improvement_test[method].dropna()
        x = np.random.normal(loc=i + 1, scale=0.05, size=len(y))
        plt.plot(x, y, 'o', alpha=0.4, markersize=4, color='blue')
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.yscale("symlog", linthresh=1)
    plt.title(
        f"Distribution of percentage improvement of {score_type} {score_name} ({direction} is better)\nrelative to baseline")
    plt.xlabel("Method")
    plt.ylabel(
        f"Percentage improvement of {score_type} {score_name}\nrelative to baseline")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    plt.savefig(f"results/analysis/Boxplot_Percentage_Improvement_{name}.png")
    plt.close()


def plot_pareto_front():
    # Example usage
    baseline = "Original"  # or however your baseline is called in df_pivot_test
    performance = pd.DataFrame(columns=['SM - Method', 'Performance'])
    performance = pd.concat([performance, pd.DataFrame(["OpenFE", 6.15])], ignore_index=True)
    performance = pd.concat([performance, pd.DataFrame(["MetaFE Random", 16.50])], ignore_index=True)
    performance = pd.concat([performance, pd.DataFrame(["MetaFE 3600", 15.146])], ignore_index=True)
    performance = pd.concat([performance, pd.DataFrame(["MetaFE 1800", 14.48])], ignore_index=True)
    performance = pd.concat([performance, pd.DataFrame(["MetaFE 300", .92])], ignore_index=True)
    performance = pd.DataFrame([
        {"SM - Method": "OpenFE", "Performance": 1.28},
        {"SM - Method": "MetaFE 7200", "Performance": 14.19},
        {"SM - Method": "MetaFE 3600", "Performance": 14.55},
        {"SM - Method": "MetaFE 1800", "Performance": 13.76},
        {"SM - Method": "MetaFE 1000", "Performance": 11.46},
        {"SM - Method": "MetaFE 500", "Performance": 9.63},
        {"SM - Method": "MetaFE 300", "Performance": 16.68},  # 24.43, 18.45
        {"SM - Method": "MetaFE 100", "Performance": -2.48}
    ])

    # === Step 2: Compute average runtime per method ===
    avg_times = pd.DataFrame([
        {"SM - Method": "OpenFE", "Runtime": 269.05},
        {"SM - Method": "MetaFE 7200", "Runtime": 5565.23},
        {"SM - Method": "MetaFE 3600", "Runtime": 3102.12},
        {"SM - Method": "MetaFE 1800", "Runtime": 1742.21},
        {"SM - Method": "MetaFE 1000", "Runtime": 994.12},
        {"SM - Method": "MetaFE 500", "Runtime": 500.20},
        {"SM - Method": "MetaFE 300", "Runtime": 300.01},
        {"SM - Method": "MetaFE 100", "Runtime": 100.01}
    ])

    # === Step 3: Merge performance + runtime ===
    merged = pd.merge(performance, avg_times, on="SM - Method", how="inner")

    # === Step 4: Identify Pareto front ===
    def is_pareto_efficient(df):
        is_efficient = np.ones(df.shape[0], dtype=bool)
        for i, (perf_i, time_i) in enumerate(zip(df["Performance"], df["Runtime"])):
            if is_efficient[i]:
                # If another point has better performance *and* lower runtime, then i is not efficient
                is_dominated = (
                        (df["Performance"] > perf_i) &
                        (df["Runtime"] < time_i)
                )
                if is_dominated.any():
                    is_efficient[i] = False
        return is_efficient

    merged["Pareto"] = is_pareto_efficient(merged)

    # === Step 5: Plot ===
    plt.figure(figsize=(12, 7))
    for i, row in merged.iterrows():
        plt.scatter(row["Runtime"], row["Performance"],
                    color='red' if row["Pareto"] else 'gray',
                    s=100, label=row["SM - Method"] if row["Pareto"] else "")

    # Connect the Pareto front
    pareto_front = merged[merged["Pareto"]].sort_values("Runtime")
    plt.plot(pareto_front["Runtime"], pareto_front["Performance"], 'r--', label="Pareto Front")

    # Annotate points
    for i, row in merged.iterrows():
        plt.text(row["Runtime"] * 1.01, row["Performance"], row["SM - Method"], fontsize=9)

    # Labels
    plt.xlabel("Average Runtime per Dataset (s)")
    plt.xscale("log")
    plt.ylabel("Average Test Error Reduction (%)")
    #plt.gca().invert_xaxis()
    plt.title("Pareto Front: Performance vs Runtime")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"results/analysis/Pareto_pandas_openfe.png")
    plt.show()


def analysis():
    baseline_col = "Original"
    result_files = glob.glob("results/Result_*.parquet")
    result_files = [f for f in result_files]
    # Get grouped data (not pivoted yet)
    df_grouped = get_data(result_files)

    # Separate by score and pivot each
    results_by_score = separate_by_score_and_pivot(df_grouped)

    print(f"Found {len(results_by_score)} unique scores: {list(results_by_score.keys())}")

    # Plot for each score separately
    for score_name, (df_pivot_val, df_pivot_val_std, df_pivot_test, df_pivot_test_std, dataset_list) in results_by_score.items():
        print(f"\nProcessing score: {score_name}")
        try:
            df_pivot_val = df_pivot_val.drop(columns=["MACFE"])
            df_pivot_val_std = df_pivot_val_std.drop(columns=["MACFE"])
            df_pivot_test = df_pivot_test.drop(columns=["MACFE"])
            df_pivot_test_std = df_pivot_test_std.drop(columns=["MACFE"])
        except KeyError:
            print("MACFE not found")
        plot_score_graph(dataset_list, df_pivot_val, df_pivot_val_std, f"Val_{score_name}")
        plot_score_graph(dataset_list, df_pivot_test, df_pivot_test_std, f"Test_{score_name}")

        plot_count_best(df_pivot_val, df_pivot_test, f"{score_name}_", score_name)
        plot_avg_percentage_impr(baseline_col, df_pivot_val, df_pivot_val_std, f"Val_{score_name}", score_name)
        plot_avg_percentage_impr(baseline_col, df_pivot_test, df_pivot_test_std, f"Test_{score_name}", score_name)

        plot_boxplot_percentage_impr(baseline_col, df_pivot_val, f"Val_{score_name}", score_name)
        plot_boxplot_percentage_impr(baseline_col, df_pivot_test, f"Test_{score_name}", score_name)

        # Without FE
        df_pivot_val_without_FE = df_pivot_val.copy()
        df_pivot_test_without_FE = df_pivot_test.copy()
        df_pivot_val_without_FE.drop(columns=["OpenFE", "MetaFE"], inplace=True, errors='ignore')
        df_pivot_test_without_FE.drop(columns=["OpenFE", "MetaFE"], inplace=True, errors='ignore')

        plot_count_best(df_pivot_val_without_FE, df_pivot_test_without_FE, f"{score_name}_without_FE_", score_name)
        plot_avg_percentage_impr(baseline_col, df_pivot_val_without_FE, df_pivot_val_std, f"Val_{score_name}_without_FE", score_name)
        plot_avg_percentage_impr(baseline_col, df_pivot_test_without_FE, df_pivot_test_std, f"Test_{score_name}_without_FE", score_name)
        plot_boxplot_percentage_impr(baseline_col, df_pivot_val_without_FE, f"Val_{score_name}_without_FE", score_name)
        plot_boxplot_percentage_impr(baseline_col, df_pivot_test_without_FE, f"Test_{score_name}_without_FE", score_name)


if __name__ == "__main__":
    analysis()
