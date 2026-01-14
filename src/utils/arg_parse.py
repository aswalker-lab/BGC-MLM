"""
Central script for all CLI argument parsing
"""

import argparse
import sys


# Central argument definitions (do not use arguments that contain '-')
ARGUMENT_MAP = {
    # Core Transformer model arguments
    "model_name":           {"type": str, "help": "Name of the model"},
    "unknown_threshold":    {"type": int, "help": "Threshold for unknown tokens"},
    "max_bgc_length":       {"type": int, "help": "Maximum BGC length"},
    "d_model":              {"type": int, "help": "Dimension of model (e.g. 512, 128)"},
    "n_layers":             {"type": int, "help": "Number of layers (e.g. 6)"},
    "heads":                {"type": int, "help": "Number of attention heads (e.g. 8)"},
    "dropout":              {"type": float, "help": "Dropout rate (e.g. 0.1)"},
    "batch_size":           {"type": int, "help": "Batch size"},
    "data_set":             {"type": str, "help": "Path to dataset file with features"},
    
    # Inference/Similarity arguments
    "model_param_file":     {"help": "Path to model parameter file"},
    "infile":               {"type": str, "help": "List of BGC ids for which to calculate similarity"},
    "token_path":           {"type": str, "help": "Path to directory with tokens"},
    "outfile":              {"type": str, "help": "Outfile name for ranking"},
    "smiles":               {"type": str, "help": "Smiles for molecule to search for"},
    
    # Metric/Regression task arguments
    "fp_file":              {"type": str, "help": "File containing fingerprints for BGC"},
    "regression_file":      {"type": str, "help": "File containing classifications/regression values for BGC"},
    "classification_file":  {"type": str, "help": "File containing classifications for BGCs"},
    "comparison_set":       {"type": str, "help": "File with molecule fp links to rank"},
    "output_index":         {"type": int, "help": "Output index"},
    
    # Visualization arguments
    "gbk":                  {"help": "Input GenBank file"},
    "acc":                  {"help": "BGC-MLM accuracy file"},
    "score":                {"help": "BGC-MLM score file"},
    
    # Optional arguments
    "seed":                 {"type": int, "default": 0, "help": "Random seed"},
    "model_output":         {"type": str, "help": "Output name for classifier"},
    "epochs":               {"type": int, "default": 50, "help": "Number of epochs"},
    "train_fraction":       {"type": float, "default": 0.9, "help": "Fraction to use for training"},
    "fp_size":              {"type": int, "default": 8192, "help": "Fingerprint size"},
    "loss_type":            {"type": str, "default": "correlation", "help": "Loss type"},
    "freeze":               {"type": int, "default": 1, "help": "Layers to freeze"},
    "fp_type":              {"type": str, "default": "morgan", "help": "Fingerprint type"},
    "num_tasks":            {"type": int, "default": 40, "help": "Number of tasks being predicted"},
    "output":               {"help": "Output HTML file", "default": "bgc_visualization.html"},
    "evaluate_only":        {"action": "store_true", "help": "Skip training and run evaluation"},
    "from_scratch":         {"action": "store_true", "help": "Train from scratch instead of loading pretrained"},
    "use_pos_weights":      {"action": "store_true", "help": "Use positive weights for loss"},
    
}

# Script to arguments mapping, adding '--' makes argument optional
SCRIPT_ARGS_MAP = {
    "classify_cluster": [  # antismash as well
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
        "classification_file",
    ],
    "fp_rank_metrics": [
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
        "comparison_set",
        "token_path",
        "--seed",
    ],
    "metric_task": [  # and metricTaskFromScratch
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
        "fp_file",
        "token_path",
        "--seed",
        "--model_output",
        "--epochs",
        "--train_fraction",
        "--fp_size",
        "--loss_type",
    ],
    "predict_sequence": [
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
    ],
    "regression_task": [    # and regressionTaskScratch, regressionTaskTest
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
        "regression_file",
        "token_path",
        "--seed",
        "--model_output",   # Unused by regressionTaskScratch
        "--freeze",         # Unused by regressionTaskScratch
    ],
    "top_k_prediction": [
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
    ],
    "bcg_mlm_train": [
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
    ],
    "classification_task_test": [  # same arguments as TaskTestOverallMetrics
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
        "classification_file",
        "token_path",
        "--seed",
    ],
    "classification_task": [    # and classificationTaskFromScratch
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
        "classification_file",
        "token_path",
        "--seed",
        "--model_output",
        "--freeze",  # freeze is unused in classificationTaskFromScratch
        "--epochs",
        "--use_pos_weights",
        "--write_metrics",
        "--train_fraction",
    ],
    "embedding_similarity": [
        "model_name",
        "model_param_file",
        "infile",
        "token_path",
        "outfile",
        "--seed",
        "--fp_size",
    ],
    "fp_rank_genome": [
        "model_name",
        "model_param_file",
        "smiles",
        "token_path",
        "outfile",
        "--seed",
        "--fp_type",
        "--fp_size",
    ],
    "predicted_fps_similarity": [
        "model_name",
        "model_param_file",
        "infile",
        "token_path",
        "outfile",
        "--seed",
        "--fp_type",
        "--fp_size",
    ],
    "regression_mask_effect": [
        "model_name",
        "model_param_file",
        "data_set",
        "token_path",
        "output_index",
        "--seed",
        "--num_tasks",
    ],
    "visualize_sequence": [
        "gbk",
        "acc",
        "score",
        "--output",
    ],
    "run_classification": [
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
        "classification_file",
        "token_path",
        "--seed",
        "--model_output", # Used for saving
        "--freeze",
        "--epochs",
        "--use_pos_weights",
        # "--write_metrics", # Implied or handled by metrics code
        "--train_fraction",
        "--evaluate_only",
        "--from_scratch",
    ],
    "run_regression": [
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
        "regression_file",
        "token_path",
        "--seed",
        "--model_output",
        "--freeze",
        "--evaluate_only",
        "--from_scratch",
    ],
    "run_metric_learning": [
        "model_name",
        "unknown_threshold",
        "max_bgc_length",
        "d_model",
        "n_layers",
        "heads",
        "dropout",
        "batch_size",
        "data_set",
        "fp_file",
        "token_path",
        "--seed",
        "--model_output",
        "--epochs",
        "--train_fraction",
        "--fp_size",
        "--loss_type",
        "--evaluate_only",
        "--from_scratch",
    ],
}


def parse_args(script_name, args_list=None, debug=False) -> argparse.Namespace:
    """
    Parses arguments for a specific script.

    Args:
        script_name (str): Identifier for the script (e.g., 'classify_cluster', 'metric_task').
        args_list (list): Optional list of arguments to parse (default: sys.argv).
        debug (bool): sets output to verbose

    Returns:
        Namespace: Parsed arguments.
    """
    # Normalize script name to lowercase
    script_name = script_name.lower()

    if script_name not in SCRIPT_ARGS_MAP:
        raise ValueError(f"Unknown script name: {script_name}")

    parser = argparse.ArgumentParser(description=f"Argument parser for {script_name}")

    # Get the list of arguments for this script
    args_for_script = SCRIPT_ARGS_MAP[script_name]

    for arg_name in args_for_script:
        if arg_name.startswith("-"):
            arg_config = ARGUMENT_MAP[arg_name.replace("-", "")].copy()
            arg_config["required"] = False
            parser.add_argument(arg_name, **arg_config)
        else:
            # Check if it's actually optional (has defaults or is flagged with --)
            arg_config = ARGUMENT_MAP[arg_name]
            if "default" in arg_config:
                arg_config["required"] = False
            parser.add_argument(arg_name, **arg_config)
            
    if debug:
        parser.print_help()

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()


if __name__ == "__main__":
    print("Running Test")
    for key in SCRIPT_ARGS_MAP.keys():
        try:
            print(f"Script: {key}")
            parse_args(key, debug=True)
        except:
            continue