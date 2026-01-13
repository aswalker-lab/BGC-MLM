import csv
import sys
from delete import parse_args

"""
Script to test the parser arguments for the assorted script to verify everything is setup correctly and working

NOTE: To use be sure to change the parser script to return the parser itself rather than directly parsing the arguments

"""

# List of all possible script names
all_scripts = [
    "classify_cluster",
    "fp_rank_metrics",
    "metric_task",
    "metric_task_scratch",
    "predict_sequence",
    "regression_task",
    "regression_task_scratch",
    "regression_task_test",
    "top_k_prediction",
    "embedding_similarity",
    "fp_rank_genome",
    "predicted_fps_similarity",
    "regression_mask_effect",
    "visualize_sequence",
]

# CSV file to save results
output_file = "script_arguments.csv"

# Collect argument information for each script
results = []

for script_name in all_scripts:
    try:
        temp_parser = parse_args(script_name=script_name)
        # Extract arguments
        arguments = []
        for action in temp_parser._actions:
            if action.dest == "help":  # Skip the help action
                continue

            arg_name = action.dest
            arg_type = None

            # Get the type
            if action.type is not None:
                arg_type = action.type.__name__

            arguments.append((arg_name, arg_type if arg_type else "None"))

        # Create row for CSV
        row = [script_name] + arguments
        results.append(row)

        print(f"✓ Processed: {script_name}")

    except Exception as e:
        print(f"✗ Error processing {script_name}: {e}")
        results.append([script_name, ("ERROR", str(e))])

# Write to CSV
try:
    with open(output_file, "w", newline="") as f:
        # First, determine the maximum number of columns needed
        max_cols = max(len(row) for row in results) if results else 1

        writer = csv.writer(f)
        writer.writerows(results)

    print(f"\n✓ Results saved to {output_file}")

except Exception as e:
    print(f"✗ Error writing to CSV: {e}")
