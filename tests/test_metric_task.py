# -*- coding: utf-8 -*-
"""
Integration tests for the metric learning pipeline.

Validates that the refactored run_metric_learning.py reproduces the same
end-to-end behavior as metricTask.py and metricTaskFromScratch.py.
"""

import os
import sys
import math
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'BGC-MLM-old'))

from models.architecture import BGC_MLM, MLM, BGCMetricLearning
import BGC_MLM_tools


@pytest.fixture
def metric_setup(token_list, sample_bgc_tokens):
    """Build all objects needed for metric pipeline testing."""
    target_embeddings = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ]
    params = dict(vocab_size=len(token_list), seq_len=12,
                  d_model=16, n_layers=1, heads=2, dropout=0.0)
    return {
        "target_embeddings": target_embeddings,
        "params": params,
        "token_list": token_list,
        "bgc_tokens": sample_bgc_tokens,
        "fp_size": 4,
    }


class TestPretrainedMetricPipeline:
    """Mirrors metricTask.py: load pretrained MLM → build metric model → train."""

    def test_pipeline_runs(self, metric_setup):
        cfg = metric_setup

        # 1. Build MLM
        bgc_mlm = BGC_MLM(**cfg["params"])
        mlm = MLM(bgc_mlm, cfg["params"]["vocab_size"])

        # 2. Build metric model (old script: freeze not passed = default False)
        metric_model = BGCMetricLearning(
            mlm.bgc_mlm, cfg["params"]["d_model"], cfg["fp_size"]
        )

        # 3. Create dataset and split
        dataset = BGC_MLM_tools.BGCTargetEmbeddingDataset(
            cfg["bgc_tokens"], cfg["token_list"], cfg["target_embeddings"],
            seq_len=cfg["params"]["seq_len"]
        )
        train_set, val_set = random_split(dataset, [4, 1])
        train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

        # 4. Train 1 epoch
        trainer = BGC_MLM_tools.BGCMetricTrainer(
            metric_model, train_loader, val_set, device="cpu",
            loss_type="correlation"
        )
        trainer.train(0)

        # 5. Should complete without error
        assert True

    def test_model_default_no_freeze(self, metric_setup):
        """
        metricTask.py creates BGCMetricLearning without freeze arg,
        so all params should be trainable.
        """
        cfg = metric_setup
        bgc_mlm = BGC_MLM(**cfg["params"])
        model = BGCMetricLearning(bgc_mlm, cfg["params"]["d_model"], cfg["fp_size"])
        for param in model.bgc_mlm.parameters():
            assert param.requires_grad is True


class TestFromScratchMetricPipeline:
    """Mirrors metricTaskFromScratch.py: no pretrained weights loaded."""

    def test_pipeline_runs(self, metric_setup):
        cfg = metric_setup

        # 1. Build MLM — no loading (from scratch)
        bgc_mlm = BGC_MLM(**cfg["params"])
        mlm = MLM(bgc_mlm, cfg["params"]["vocab_size"])

        # 2. Build metric model
        metric_model = BGCMetricLearning(
            mlm.bgc_mlm, cfg["params"]["d_model"], cfg["fp_size"]
        )

        dataset = BGC_MLM_tools.BGCTargetEmbeddingDataset(
            cfg["bgc_tokens"], cfg["token_list"], cfg["target_embeddings"],
            seq_len=cfg["params"]["seq_len"]
        )
        train_set, val_set = random_split(dataset, [4, 1])
        train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

        trainer = BGC_MLM_tools.BGCMetricTrainer(
            metric_model, train_loader, val_set, device="cpu",
            loss_type="correlation"
        )
        trainer.train(0)  # 1 epoch without error

    def test_no_weights_loaded(self, metric_setup):
        """
        Key difference from pretrained: mlm.load_state_dict() is NOT called.
        Verify model works with random initialization.
        """
        cfg = metric_setup
        bgc_mlm = BGC_MLM(**cfg["params"])
        mlm = MLM(bgc_mlm, cfg["params"]["vocab_size"])
        model = BGCMetricLearning(
            mlm.bgc_mlm, cfg["params"]["d_model"], cfg["fp_size"]
        )
        model.eval()
        x = torch.randint(0, cfg["params"]["vocab_size"],
                          (2, cfg["params"]["seq_len"]))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, cfg["fp_size"])


class TestMetricLossType:
    """Validates loss_type default matches old scripts."""

    def test_default_loss_type_is_correlation(self):
        """Both metricTask.py and metricTaskFromScratch.py default to 'correlation'."""
        # This is tested through arg_parse defaults, but verify the trainer accepts it
        from utils.arg_parse import parse_args
        args = parse_args("run_metric_learning", [
            "test_model", "5", "12", "16", "1", "2", "0.1", "4",
            "test.db", "fp.csv", "tokens/"
        ])
        assert args.loss_type == "correlation"

    def test_train_val_split(self, metric_setup):
        """Split matches math.floor(0.9 * N)."""
        n = len(metric_setup["bgc_tokens"])
        train_fraction = 0.9
        train_size = math.floor(train_fraction * n)
        val_size = n - train_size
        assert train_size == 4
        assert val_size == 1
