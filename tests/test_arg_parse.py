# -*- coding: utf-8 -*-
"""
Tests for src/utils/arg_parse.py — argument parsing for all script variants.

Verifies that the refactored central argument parser accepts the same
positional and optional arguments as the individual old scripts.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.arg_parse import parse_args


# Shared base positional args used by most scripts
BASE_POSITIONAL = [
    "test_model",       # model_name
    "5",                # unknown_threshold
    "12",               # max_bgc_length
    "16",               # d_model
    "1",                # n_layers
    "2",                # heads
    "0.1",              # dropout
    "4",                # batch_size
    "test_dataset.db",  # data_set
]


class TestClassificationArgs:

    def test_run_classification_positional(self):
        """run_classification accepts all positional args matching old classificationTask.py."""
        args_list = BASE_POSITIONAL + [
            "class_file.csv",   # classification_file
            "tokens/",          # token_path
        ]
        args = parse_args("run_classification", args_list)
        assert args.model_name == "test_model"
        assert args.unknown_threshold == 5
        assert args.max_bgc_length == 12
        assert args.d_model == 16
        assert args.n_layers == 1
        assert args.heads == 2
        assert args.dropout == 0.1
        assert args.batch_size == 4
        assert args.data_set == "test_dataset.db"
        assert args.classification_file == "class_file.csv"
        assert args.token_path == "tokens/"

    def test_from_scratch_flag(self):
        """--from_scratch flag mirrors classificationTaskFromScratch behavior."""
        args_list = BASE_POSITIONAL + [
            "class_file.csv",
            "tokens/",
            "--from_scratch",
        ]
        args = parse_args("run_classification", args_list)
        assert args.from_scratch is True

    def test_evaluate_only_flag(self):
        """--evaluate_only flag mirrors classificationTaskTest behavior."""
        args_list = BASE_POSITIONAL + [
            "class_file.csv",
            "tokens/",
            "--evaluate_only",
        ]
        args = parse_args("run_classification", args_list)
        assert args.evaluate_only is True

    def test_optional_args_defaults(self):
        """Optional args have sensible defaults matching old scripts."""
        args_list = BASE_POSITIONAL + ["class_file.csv", "tokens/"]
        args = parse_args("run_classification", args_list)
        assert args.seed == 0
        assert args.epochs == 50
        assert args.train_fraction == 0.9

    def test_freeze_default(self):
        """Default freeze=1 matches old classificationTask.py default."""
        args_list = BASE_POSITIONAL + ["class_file.csv", "tokens/"]
        args = parse_args("run_classification", args_list)
        assert args.freeze == 1


class TestRegressionArgs:

    def test_run_regression_positional(self):
        """run_regression accepts the same positional args as old regressionTask.py."""
        args_list = BASE_POSITIONAL + [
            "regression_file.csv",
            "tokens/",
        ]
        args = parse_args("run_regression", args_list)
        assert args.regression_file == "regression_file.csv"
        assert args.token_path == "tokens/"

    def test_from_scratch_and_evaluate(self):
        """Both flags are available for regression."""
        args_list = BASE_POSITIONAL + [
            "regression_file.csv",
            "tokens/",
            "--from_scratch",
        ]
        args = parse_args("run_regression", args_list)
        assert args.from_scratch is True

        args_list2 = BASE_POSITIONAL + [
            "regression_file.csv",
            "tokens/",
            "--evaluate_only",
        ]
        args2 = parse_args("run_regression", args_list2)
        assert args2.evaluate_only is True


class TestMetricLearningArgs:

    def test_run_metric_learning_positional(self):
        """run_metric_learning accepts the same args as old metricTask.py."""
        args_list = BASE_POSITIONAL + [
            "fp_file.csv",
            "tokens/",
        ]
        args = parse_args("run_metric_learning", args_list)
        assert args.fp_file == "fp_file.csv"
        assert args.token_path == "tokens/"

    def test_optional_defaults(self):
        """Default fp_size=8192 and loss_type='correlation' match old scripts."""
        args_list = BASE_POSITIONAL + ["fp_file.csv", "tokens/"]
        args = parse_args("run_metric_learning", args_list)
        assert args.fp_size == 8192
        assert args.loss_type == "correlation"
        assert args.train_fraction == 0.9
        assert args.epochs == 50


class TestErrorHandling:

    def test_unknown_script_raises(self):
        """Unrecognised script name must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown script name"):
            parse_args("does_not_exist", [])
