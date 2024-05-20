import json
import os
import pandas as pd
from constants import (
    TASK_NAMES,
    SELECTIVE_ANNOTATION_METHODS,
    QUERYLESS_SUBMODLIB_FUNCTIONS_TWO_SEQUENCE,
)

OUTPUTS_DIR = "outputs"


# Function to parse the accuracy from a result_summary.txt file
def parse_accuracy(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        accuracy = content.split("accuracy is: ")[1].strip()
    return float(accuracy)


rows = []
soft_rows = []
# Iterate over every combination of task and method
for model_name in os.listdir(OUTPUTS_DIR):
    for task in TASK_NAMES:
        for method in SELECTIVE_ANNOTATION_METHODS:
            file_path = os.path.join(
                OUTPUTS_DIR, model_name, task, method, "result_summary.txt"
            )
            json_file_path = file_path.replace(".txt", ".json")
            metadata = {
                "model_name": model_name,
                "task": task,
                "method": method,
            }

            if os.path.exists(file_path):
                accuracy = parse_accuracy(file_path)
                rows.append({**metadata, "accuracy": accuracy})
            elif os.path.exists(json_file_path):
                with open(json_file_path) as f:
                    results = json.load(f)
                soft_rows.append({**metadata, **results})
            else:
                # print(f"File not found: {file_path}")
                ...

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(rows)

# Pivot the DataFrame
pivot_df = df.pivot(index=["model_name", "method"], columns="task", values="accuracy")

# Add an 'average' column which is the row average
pivot_df["average"] = pivot_df.mean(axis=1)

# Sort the DataFrame based on the 'average' column in ascending order
pivot_df_sorted = pivot_df.sort_values(by="average")

# Save the sorted pivoted DataFrame to a CSV file
file_name = f"results_hard_summary.csv"
pivot_df_sorted.to_csv(file_name)

print(f"Sorted CSV file has been created as {file_name}.")

soft_df = pd.DataFrame(soft_rows)
# metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum", "accuracy"]

file_name = f"soft_results_summary.csv"
soft_df.to_csv(f"results_soft_summary.csv", index=False)

print(f"Sorted CSV file has been created as {file_name}.")

df = pd.read_csv(f"results_hard_summary.csv")

# Filter the DataFrame based on the method
df = df[df["method"].isin(QUERYLESS_SUBMODLIB_FUNCTIONS_TWO_SEQUENCE)]

# Split the 'method' column into 'first_method' and 'second_method'
split_methods = df["method"].str.split("-", expand=True)
df["first_method"] = split_methods[0]
df["second_method"] = split_methods[1]

# Pivot the DataFrame
pivoted_df = df.pivot(index="second_method", columns="first_method", values="average")

# Print the pivoted DataFrame
pivoted_df.to_csv(f"two_method_results_summary.csv")
