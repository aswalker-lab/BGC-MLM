# -*- coding: utf-8 -*-
"""
Tests for model architecture classes in src/models/architecture.py.

Verifies output shapes, freeze behavior, and weight sharing — matching
the construction patterns used across all 8 old scripts.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.architecture import BGC_MLM, MLM, BGCMultiLabelClassifier, BGCRegression, BGCMetricLearning


class TestBGCMLM:

    def test_output_shape(self, small_model_params):
        """BGC_MLM forward returns (batch, seq_len, d_model)."""
        model = BGC_MLM(**small_model_params)
        model.eval()
        batch_size = 3
        x = torch.randint(0, small_model_params["vocab_size"],
                          (batch_size, small_model_params["seq_len"]))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, small_model_params["seq_len"],
                             small_model_params["d_model"])


class TestMLM:

    def test_output_shape(self, small_model_params):
        """MLM forward returns (batch, seq_len, vocab_size)."""
        bgc_mlm = BGC_MLM(**small_model_params)
        mlm = MLM(bgc_mlm, small_model_params["vocab_size"])
        mlm.eval()
        batch_size = 2
        x = torch.randint(0, small_model_params["vocab_size"],
                          (batch_size, small_model_params["seq_len"]))
        with torch.no_grad():
            out = mlm(x)
        assert out.shape == (batch_size, small_model_params["seq_len"],
                             small_model_params["vocab_size"])


class TestBGCMultiLabelClassifier:

    def test_output_shape(self, small_model_params):
        """Classifier returns (batch, n_tasks)."""
        bgc_mlm = BGC_MLM(**small_model_params)
        n_tasks = 3
        classifier = BGCMultiLabelClassifier(bgc_mlm, small_model_params["d_model"], n_tasks)
        classifier.eval()
        batch_size = 2
        x = torch.randint(0, small_model_params["vocab_size"],
                          (batch_size, small_model_params["seq_len"]))
        with torch.no_grad():
            out = classifier(x)
        assert out.shape == (batch_size, n_tasks)

    def test_freeze_true(self, small_model_params):
        """When freeze=True, bgc_mlm parameters have requires_grad=False."""
        bgc_mlm = BGC_MLM(**small_model_params)
        classifier = BGCMultiLabelClassifier(bgc_mlm, small_model_params["d_model"], 3, freeze=True)
        for param in classifier.bgc_mlm.parameters():
            assert param.requires_grad is False

    def test_freeze_false(self, small_model_params):
        """When freeze=False (default), bgc_mlm parameters have requires_grad=True."""
        bgc_mlm = BGC_MLM(**small_model_params)
        classifier = BGCMultiLabelClassifier(bgc_mlm, small_model_params["d_model"], 3, freeze=False)
        for param in classifier.bgc_mlm.parameters():
            assert param.requires_grad is True

    def test_shared_weights_with_mlm(self, small_model_params):
        """
        BGCMultiLabelClassifier(mlm.bgc_mlm, ...) shares the bgc_mlm reference
        with the MLM object. This is critical for the pretrained loading pattern
        used in classificationTask.py: loading weights into mlm also updates
        the classifier's backbone.
        """
        bgc_mlm = BGC_MLM(**small_model_params)
        mlm = MLM(bgc_mlm, small_model_params["vocab_size"])
        classifier = BGCMultiLabelClassifier(mlm.bgc_mlm, small_model_params["d_model"], 3)

        # They should be the exact same object in memory
        assert classifier.bgc_mlm is mlm.bgc_mlm

        # Verify that modifying mlm.bgc_mlm affects classifier
        with torch.no_grad():
            for param in mlm.bgc_mlm.parameters():
                param.fill_(42.0)
                break  # just check first param
        first_param_classifier = next(classifier.bgc_mlm.parameters())
        assert torch.all(first_param_classifier == 42.0)


class TestBGCRegression:

    def test_output_shape(self, small_model_params):
        """Regression model returns (batch, n_tasks)."""
        bgc_mlm = BGC_MLM(**small_model_params)
        n_tasks = 2
        model = BGCRegression(bgc_mlm, small_model_params["d_model"], n_tasks)
        model.eval()
        batch_size = 2
        x = torch.randint(0, small_model_params["vocab_size"],
                          (batch_size, small_model_params["seq_len"]))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, n_tasks)

    def test_freeze_behavior(self, small_model_params):
        """freeze=True locks bgc_mlm params (matching regressionTask.py freeze logic)."""
        bgc_mlm = BGC_MLM(**small_model_params)
        model = BGCRegression(bgc_mlm, small_model_params["d_model"], 2, freeze=True)
        for param in model.bgc_mlm.parameters():
            assert param.requires_grad is False


class TestBGCMetricLearning:

    def test_output_shape(self, small_model_params):
        """Metric learning model returns (batch, d_fp)."""
        bgc_mlm = BGC_MLM(**small_model_params)
        d_fp = 8
        model = BGCMetricLearning(bgc_mlm, small_model_params["d_model"], d_fp)
        model.eval()
        batch_size = 2
        x = torch.randint(0, small_model_params["vocab_size"],
                          (batch_size, small_model_params["seq_len"]))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, d_fp)

    def test_default_no_freeze(self, small_model_params):
        """
        Default freeze=False matches old metricTask.py behavior where
        BGCMetricLearning was created without freeze argument.
        """
        bgc_mlm = BGC_MLM(**small_model_params)
        model = BGCMetricLearning(bgc_mlm, small_model_params["d_model"], 8)
        for param in model.bgc_mlm.parameters():
            assert param.requires_grad is True
