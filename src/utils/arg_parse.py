"""
Central script for all CLI argument parsing
"""

import argparse
import sys


def add_core_model_args(parser):
    """
    Adds core Transformer model arguments used by training/prediction scripts.
    Positional arguments:
      model_name
      unknown_threshold
      max_bgc_length
      d_model
      n_layers
      heads
      dropout
      batch_size
      data_set
    """
    parser.add_argument("model_name", help="Name of the model")
    parser.add_argument(
        "unknown_threshold", type=int, help="Threshold for unknown tokens"
    )
    parser.add_argument("max_bgc_length", type=int, help="Maximum BGC length")
    parser.add_argument("d_model", type=int, help="Dimension of model (e.g. 512, 128)")
    parser.add_argument("n_layers", type=int, help="Number of layers (e.g. 6)")
    parser.add_argument("heads", type=int, help="Number of attention heads (e.g. 8)")
    parser.add_argument("dropout", type=float, help="Dropout rate (e.g. 0.1)")
    parser.add_argument("batch_size", type=int, help="Batch size")
    parser.add_argument("data_set", type=str, help="Path to dataset file with features")


def add_inference_base_args(parser):
    """
    Adds arguments for scripts that load a pretrained model by parameter file.
    Positional arguments:
      model_name
      model_param_file
    """
    parser.add_argument("model_name", help="Name of the model")
    parser.add_argument("model_param_file", help="Path to model parameter file")


def parse_args(script_name, args_list=None):
    """
    Parses arguments for a specific script.

    Args:
        script_name (str): Identifier for the script (e.g., 'classify_cluster', 'metric_task').
        args_list (list): Optional list of arguments to parse (default: sys.argv).

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=f"Argument parser for {script_name}")

    # --- Group 1: Transformer Model Scripts ---
    # Scripts: classify cluster, fpRankMetrics, MetricTask, predictSequence,
    #          regressionTask, TopKPrediction

    transformer_scripts = {
        "classify_cluster",
        "fp_rank_metrics",
        "metric_task",
        "metric_task_scratch",
        "predict_sequence",
        "regression_task",
        "regression_task_scratch",
        "regression_task_test",
        "top_k_prediction",
    }

    if script_name in transformer_scripts:
        add_core_model_args(parser)

        # Script-specific positional args and flags
        if script_name == "classify_cluster":
            parser.add_argument(
                "classification_file",
                type=str,
                help="File containing classifications for BGCs",
            )

        elif script_name == "fp_rank_metrics":
            parser.add_argument(
                "comparison_set", type=str, help="File with molecule fp links to rank"
            )
            parser.add_argument(
                "token_path", type=str, help="Path to directory with tokens"
            )
            parser.add_argument("--seed", type=int, default=0, help="Random seed")

        elif script_name in ["metric_task", "metric_task_scratch"]:
            parser.add_argument(
                "fp_file", type=str, help="File containing fingerprints for BGC"
            )
            parser.add_argument(
                "token_path", type=str, help="Path to directory with tokens"
            )
            parser.add_argument("--seed", type=int, default=0, help="Random seed")
            parser.add_argument(
                "--model_output",
                type=str,
                default="metric",
                help="Output name for classifier",
            )
            parser.add_argument(
                "--epochs", type=int, default=50, help="Number of epochs"
            )
            parser.add_argument(
                "--train_fraction",
                type=float,
                default=0.9,
                help="Fraction to use for training",
            )
            parser.add_argument(
                "--fp_size", type=int, default=8192, help="Fingerprint size"
            )
            parser.add_argument(
                "--loss_type", type=str, default="correlation", help="Loss type"
            )

        elif script_name in [
            "regression_task",
            "regression_task_scratch",
            "regression_task_test",
        ]:
            parser.add_argument(
                "regression_file",
                type=str,
                help="File containing classifications/regression values for BGC",
            )
            parser.add_argument(
                "token_path", type=str, help="Path to directory with tokens"
            )
            parser.add_argument("--seed", type=int, default=0, help="Random seed")

            # Optional args specific to regressionTask (not explicitly in Scratch/Test versions in original file, but safe to add as optional)
            help_text_model_output = (
                "Output name for classifier (Used by regressionTask)"
            )
            help_text_freeze = "Layers to freeze (Used by regressionTask)"

            parser.add_argument(
                "--model_output",
                type=str,
                default="regression",
                required=False,
                help=help_text_model_output,
            )
            parser.add_argument(
                "--freeze", type=int, default=1, required=False, help=help_text_freeze
            )

        elif script_name in ["predict_sequence", "top_k_prediction"]:
            # These scripts had no extra args beyond the core set in the original file
            pass

    # --- Group 2: Inference/Similarity Scripts ---
    # Scripts: embedding similarity, fpRankGenome, PredictedFPSSimilarity, regressionTaskMaskEffect

    elif script_name == "embedding_similarity":
        add_inference_base_args(parser)
        parser.add_argument(
            "infile", type=str, help="List of BGC ids for which to calculate similarity"
        )
        parser.add_argument(
            "token_path", type=str, help="Path to directory with tokens from genome"
        )
        parser.add_argument(
            "outfile", type=str, help="Outfile prefix for similarity calculation"
        )
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument(
            "--fp_size",
            type=int,
            default=8192,
            help="Fraction to use for training (or fp size)",
        )

    elif script_name == "fp_rank_genome":
        add_inference_base_args(parser)
        parser.add_argument(
            "smiles", type=str, help="Smiles for molecule to search for"
        )
        parser.add_argument(
            "token_path", type=str, help="Path to directory with tokens from genome"
        )
        parser.add_argument("outfile", type=str, help="Outfile name for ranking")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument(
            "--fp_type", type=str, default="morgan", help="Fingerprint type"
        )
        parser.add_argument(
            "--fp_size", type=int, default=2048, help="Fingerprint length"
        )

    elif script_name == "predicted_fps_similarity":
        add_inference_base_args(parser)
        parser.add_argument(
            "infile", type=str, help="List of BGC ids for which to calculate similarity"
        )
        parser.add_argument(
            "token_path", type=str, help="Path to directory with tokens from genome"
        )
        parser.add_argument(
            "outfile", type=str, help="Outfile prefix for similarity calculation"
        )
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument(
            "--fp_type", type=str, default="morgan", help="Fingerprint type"
        )
        parser.add_argument(
            "--fp_size", type=int, default=2048, help="Fingerprint length"
        )

    elif script_name == "regression_mask_effect":
        add_inference_base_args(parser)
        parser.add_argument(
            "data_set", type=str, help="Path to dataset file with features"
        )
        parser.add_argument(
            "token_path", type=str, help="Path to directory with tokens"
        )
        parser.add_argument("output_index", type=int, help="Output index")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument(
            "--num_tasks", type=int, default=40, help="Number of tasks being predicted"
        )

    # --- Group 3: Visualization ---
    elif script_name == "visualize_sequence":
        parser.add_argument("gbk", help="Input GenBank file")
        parser.add_argument("acc", help="BGC-MLM accuracy file")
        parser.add_argument("score", help="BGC-MLM score file")
        parser.add_argument(
            "-o", "--output", help="Output HTML file", default="bgc_visualization.html"
        )

    else:
        raise ValueError(f"Unknown script name: {script_name}")

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()


# Example usage (commented out):
# if __name__ == "__main__":
#     # Test with specific script
#     # args = parse_args('regression_task')
#     pass
