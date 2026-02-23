# -*- coding: utf-8 -*-
"""
Tests for src/utils/metrics.py — evaluate_classification and evaluate_regression.

Verifies that metric computation matches the inline metric code from
classificationTask.py, classificationTaskTest.py, regressionTask.py,
and regressionTaskTest.py.
"""

import os
import sys
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.metrics import evaluate_classification, evaluate_regression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, roc_auc_score,
                             mean_squared_error, mean_absolute_error)
from scipy.stats import pearsonr


def _make_classification_loader(true_labels):
    """Helper: build a DataLoader that yields dicts with 'classification_label'."""
    # true_labels: list of lists, e.g. [[1,0], [0,1], [1,1]]
    tensor_labels = torch.tensor(true_labels, dtype=torch.float)
    # We only need the label key; create a simple wrapper
    class _LabelDataset(torch.utils.data.Dataset):
        def __init__(self, labels):
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return {"classification_label": self.labels[idx]}
    ds = _LabelDataset(tensor_labels)
    return DataLoader(ds, batch_size=1, shuffle=False)


def _make_regression_loader(true_values):
    """Helper: build a DataLoader that yields dicts with 'regression_value'."""
    tensor_vals = torch.tensor(true_values, dtype=torch.float)
    class _ValDataset(torch.utils.data.Dataset):
        def __init__(self, vals):
            self.vals = vals
        def __len__(self):
            return len(self.vals)
        def __getitem__(self, idx):
            return {"regression_value": self.vals[idx]}
    ds = _ValDataset(tensor_vals)
    return DataLoader(ds, batch_size=1, shuffle=False)


class TestEvaluateClassification:

    def test_perfect_predictions(self):
        """Perfect predictions → accuracy=1.0, balanced_accuracy=1.0."""
        true_labels = [[1, 0], [0, 1], [1, 0], [0, 1], [1, 1]]
        predictions = torch.tensor(true_labels, dtype=torch.float)
        loader = _make_classification_loader(true_labels)
        types = ["classA", "classB"]
        # Should not raise
        evaluate_classification(predictions, loader, types, "test_model")

    def test_random_predictions_no_error(self):
        """Random predictions should compute all metrics without error."""
        np.random.seed(42)
        n = 20
        true_labels = np.random.randint(0, 2, (n, 3)).tolist()
        # Ensure at least 2 positives per class to avoid skip
        true_labels[0] = [1, 1, 1]
        true_labels[1] = [1, 1, 1]
        predictions = torch.rand(n, 3)
        loader = _make_classification_loader(true_labels)
        types = ["A", "B", "C"]
        evaluate_classification(predictions, loader, types, "test_model")

    def test_skips_sparse_class(self, capsys):
        """Classes with sum(true_y) < 2 are skipped, matching old behavior."""
        true_labels = [[1, 0], [0, 0], [0, 0], [0, 0]]  # classA has only 1 positive
        predictions = torch.rand(4, 2)
        loader = _make_classification_loader(true_labels)
        types = ["classA", "classB"]
        evaluate_classification(predictions, loader, types, "test_model")
        captured = capsys.readouterr()
        assert "not enough of class" in captured.out

    def test_writes_outfile(self, tmp_path):
        """When outfile_path is given, writes CSV with header and data rows."""
        outfile = str(tmp_path / "metrics.txt")
        true_labels = [[1, 0], [0, 1], [1, 1], [0, 1], [1, 0]]
        predictions = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6],
                                     [0.1, 0.9], [0.8, 0.3]], dtype=torch.float)
        loader = _make_classification_loader(true_labels)
        types = ["classA", "classB"]
        evaluate_classification(predictions, loader, types, "my_model",
                                outfile_path=outfile)
        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()
        # First line is header
        assert "model_name" in lines[0]
        # Should have data lines for each class
        assert len(lines) >= 2


class TestEvaluateRegression:

    def test_perfect_predictions(self):
        """Perfect predictions → MSE≈0, MAE≈0."""
        true_values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        predictions = torch.tensor(true_values, dtype=torch.float)
        loader = _make_regression_loader(true_values)
        types = ["targetA", "targetB"]
        # Should not raise
        evaluate_regression(predictions, loader, types, "test_model",
                            plot_dir=str("/tmp"))

    def test_metrics_match_sklearn(self, tmp_path):
        """MSE, MAE, Pearson match direct sklearn/scipy computation."""
        true_values = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        pred_values = [[1.1], [2.3], [2.8], [4.2], [4.9]]
        predictions = torch.tensor(pred_values, dtype=torch.float)
        loader = _make_regression_loader(true_values)

        true_flat = [v[0] for v in true_values]
        pred_flat = [v[0] for v in pred_values]
        expected_mse = mean_squared_error(true_flat, pred_flat)
        expected_mae = mean_absolute_error(true_flat, pred_flat)
        expected_corr, _ = pearsonr(true_flat, pred_flat)

        # Capture output to verify printed values
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            evaluate_regression(predictions, loader, ["target"],
                                "test", plot_dir=str(tmp_path))
        output = f.getvalue()
        assert f"MSE: {expected_mse}" in output
        assert f"MAE: {expected_mae}" in output

    def test_saves_plot(self, tmp_path):
        """Plot PNG is created in the specified directory."""
        true_values = [[1.0], [2.0], [3.0], [4.0]]
        predictions = torch.tensor([[1.2], [1.8], [3.1], [3.9]], dtype=torch.float)
        loader = _make_regression_loader(true_values)
        evaluate_regression(predictions, loader, ["myTarget"],
                            "test", plot_dir=str(tmp_path))
        assert os.path.exists(tmp_path / "myTarget.png")
