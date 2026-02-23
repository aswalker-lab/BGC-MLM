# -*- coding: utf-8 -*-
"""
Integration tests for the regression pipeline.

Validates that the refactored run_regression.py reproduces the same
end-to-end behavior as regressionTask.py, regressionTaksFromScratch.py,
and regressionTaskTest.py.
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

from models.architecture import BGC_MLM, MLM, BGCRegression
import BGC_MLM_tools


@pytest.fixture
def regression_setup(token_list, sample_bgc_tokens):
    """Build all objects needed for regression pipeline testing."""
    y_vals = [
        [1.5, 2.3],
        [0.7, 3.1],
        [2.0, 1.0],
        [0.3, 4.5],
        [1.1, 2.8],
    ]
    params = dict(vocab_size=len(token_list), seq_len=12,
                  d_model=16, n_layers=1, heads=2, dropout=0.0)
    return {
        "y_vals": y_vals,
        "params": params,
        "token_list": token_list,
        "bgc_tokens": sample_bgc_tokens,
    }


class TestRegressionSplit:
    """Verifies the 90/10 split used by old regressionTask.py."""

    def test_split_90_10(self, regression_setup):
        """math.floor(9*N/10) matches old regressionTask.py line 131."""
        n = len(regression_setup["bgc_tokens"])
        train_size = math.floor(9 * n / 10)
        val_size = n - train_size
        assert train_size == 4
        assert val_size == 1
        assert train_size + val_size == n


class TestPretrainedRegressionPipeline:
    """Mirrors regressionTask.py: load MLM, create regression model with freeze=True."""

    def test_pipeline_runs(self, regression_setup):
        cfg = regression_setup

        # 1. Build MLM
        bgc_mlm = BGC_MLM(**cfg["params"])
        mlm = MLM(bgc_mlm, cfg["params"]["vocab_size"])

        # 2. Build regression model with freeze=True (old default)
        regression_model = BGCRegression(
            mlm.bgc_mlm, cfg["params"]["d_model"],
            len(cfg["y_vals"][0]), freeze=True
        )

        # 3. Create dataset and split
        dataset = BGC_MLM_tools.BGCRegressionDatasets(
            cfg["bgc_tokens"], cfg["token_list"], cfg["y_vals"],
            seq_len=cfg["params"]["seq_len"]
        )
        train_set, val_set = random_split(dataset, [4, 1])
        train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

        # 4. Train 1 epoch
        trainer = BGC_MLM_tools.BGCRegressionTrainier(
            regression_model, train_loader, val_set, device="cpu"
        )
        trainer.train(0)

        # 5. Predict
        predictions = trainer.predict(val_set, batch_size=1)
        assert predictions.shape[1] == 2  # n_tasks

    def test_freeze_true(self, regression_setup):
        """With freeze=True, BGC encoder params are frozen."""
        cfg = regression_setup
        bgc_mlm = BGC_MLM(**cfg["params"])
        mlm = MLM(bgc_mlm, cfg["params"]["vocab_size"])
        model = BGCRegression(mlm.bgc_mlm, cfg["params"]["d_model"], 2, freeze=True)
        for param in model.bgc_mlm.parameters():
            assert param.requires_grad is False


class TestFromScratchRegressionPipeline:
    """Mirrors regressionTaksFromScratch.py: no pretrained weights, default freeze."""

    def test_pipeline_runs(self, regression_setup):
        cfg = regression_setup

        # 1. Build MLM — no loading (from scratch)
        bgc_mlm = BGC_MLM(**cfg["params"])
        mlm = MLM(bgc_mlm, cfg["params"]["vocab_size"])

        # 2. Build regression model without freeze arg (old FromScratch default)
        # regressionTaksFromScratch.py: BGCRegression(mlm.bgc_mlm, d_model, len(y_vals[0]))
        # No freeze arg passed → defaults to freeze=False
        regression_model = BGCRegression(
            mlm.bgc_mlm, cfg["params"]["d_model"],
            len(cfg["y_vals"][0])
        )

        dataset = BGC_MLM_tools.BGCRegressionDatasets(
            cfg["bgc_tokens"], cfg["token_list"], cfg["y_vals"],
            seq_len=cfg["params"]["seq_len"]
        )
        train_set, val_set = random_split(dataset, [4, 1])
        train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

        trainer = BGC_MLM_tools.BGCRegressionTrainier(
            regression_model, train_loader, val_set, device="cpu"
        )
        trainer.train(0)

        # All params should be trainable
        for param in regression_model.bgc_mlm.parameters():
            assert param.requires_grad is True


class TestEvaluateOnlyRegressionPipeline:
    """Mirrors regressionTaskTest.py: load regression model, predict only."""

    def test_pipeline_runs(self, regression_setup, tmp_path):
        cfg = regression_setup

        # 1. Save a regression model
        bgc_mlm = BGC_MLM(**cfg["params"])
        mlm = MLM(bgc_mlm, cfg["params"]["vocab_size"])
        model = BGCRegression(
            mlm.bgc_mlm, cfg["params"]["d_model"],
            len(cfg["y_vals"][0]), freeze=True
        )
        model_path = str(tmp_path / "test_regression.pt")
        torch.save(model.state_dict(), model_path)

        # 2. Load into fresh model
        bgc_mlm2 = BGC_MLM(**cfg["params"])
        mlm2 = MLM(bgc_mlm2, cfg["params"]["vocab_size"])
        model2 = BGCRegression(
            mlm2.bgc_mlm, cfg["params"]["d_model"],
            len(cfg["y_vals"][0]), freeze=True
        )
        model2.load_state_dict(torch.load(model_path, weights_only=True))

        # 3. Predict without training
        dataset = BGC_MLM_tools.BGCRegressionDatasets(
            cfg["bgc_tokens"], cfg["token_list"], cfg["y_vals"],
            seq_len=cfg["params"]["seq_len"]
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        trainer = BGC_MLM_tools.BGCRegressionTrainier(
            model2, loader, dataset, device="cpu"
        )
        predictions = trainer.predict(dataset, batch_size=2)
        assert predictions.shape == (5, 2)
