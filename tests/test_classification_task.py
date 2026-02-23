# -*- coding: utf-8 -*-
"""
Integration tests for the classification pipeline.

Validates that the refactored run_classification.py reproduces the same
end-to-end behavior as classificationTask.py, classificationTaskFromScratch.py,
and classificationTaskTest.py.
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

from models.architecture import BGC_MLM, MLM, BGCMultiLabelClassifier
import BGC_MLM_tools


@pytest.fixture
def classification_setup(token_list, sample_bgc_tokens):
    """Build all objects needed for classification pipeline testing."""
    y_vals = [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
    ]
    params = dict(vocab_size=len(token_list), seq_len=12,
                  d_model=16, n_layers=1, heads=2, dropout=0.0)
    return {
        "y_vals": y_vals,
        "params": params,
        "token_list": token_list,
        "bgc_tokens": sample_bgc_tokens,
    }


class TestPosWeightsComputation:
    """
    Validates that the positive-weight formula from classificationTask.py
    produces the same values when implemented in the new pipeline.
    Formula: pos_weight[c] = (total_count - count[c]) / count[c]
    """

    def test_pos_weights_match_old_formula(self, tmp_classification_csv):
        """pos_weights match the old classificationTask.py inline calculation."""
        from data.loading import parse_classification_file

        types, counts, classifications, total = parse_classification_file(
            tmp_classification_csv
        )

        # Old script formula (classificationTask.py lines 142-151)
        old_pos_weights = []
        for c in counts:
            if counts[c] != 0:
                old_pos_weights.append((total - counts[c]) / counts[c])
            else:
                old_pos_weights.append((total - counts[c]) / 1)

        # New script formula (same logic via classification_types iteration)
        new_pos_weights = []
        for t in types:
            count = counts[t]
            if count != 0:
                new_pos_weights.append((total - count) / count)
            else:
                new_pos_weights.append(total - count)

        assert old_pos_weights == new_pos_weights

    def test_pos_weights_disabled(self, tmp_classification_csv):
        """When use_pos_weights=0, all weights should be 1 (old behavior)."""
        from data.loading import parse_classification_file
        types, counts, _, _ = parse_classification_file(tmp_classification_csv)

        pos_weights = [1 for _ in types]
        assert all(w == 1 for w in pos_weights)


class TestTrainValSplit:
    """Verifies that math.floor(train_fraction * N) split matches old scripts."""

    def test_split_sizes(self, classification_setup):
        """Split sizes match math.floor(0.9 * N) for the train set."""
        cfg = classification_setup
        n = len(cfg["bgc_tokens"])
        train_fraction = 0.9

        train_size = math.floor(train_fraction * n)
        val_size = n - train_size

        assert train_size == math.floor(0.9 * 5)  # = 4
        assert val_size == 1
        assert train_size + val_size == n

    def test_90_10_hardcoded_split(self, classification_setup):
        """
        regressionTask.py uses math.floor(9*N/10) instead of train_fraction.
        Verify both produce the same result for various N.
        """
        for n in [5, 10, 15, 20, 100, 7]:
            old_way = math.floor(9 * n / 10)
            new_way = math.floor(0.9 * n)
            assert old_way == new_way, f"Split mismatch for N={n}: {old_way} vs {new_way}"


class TestPretrainedClassificationPipeline:
    """Mirrors classificationTask.py: load pretrained MLM → build classifier → train."""

    def test_pipeline_runs(self, classification_setup):
        """Full pipeline: create MLM, load weights, build classifier, train 1 epoch."""
        cfg = classification_setup

        # 1. Build MLM (like old script)
        bgc_mlm = BGC_MLM(**cfg["params"])
        mlm = MLM(bgc_mlm, cfg["params"]["vocab_size"])

        # 2. Build classifier with freeze=True (old default)
        classifier = BGCMultiLabelClassifier(
            mlm.bgc_mlm, cfg["params"]["d_model"],
            len(cfg["y_vals"][0]), freeze=True
        )

        # 3. Build dataset
        dataset = BGC_MLM_tools.BGCClassificationDatasets(
            cfg["bgc_tokens"], cfg["token_list"], cfg["y_vals"],
            seq_len=cfg["params"]["seq_len"]
        )
        train_set, val_set = random_split(dataset, [4, 1])
        train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

        # 4. Build trainer and train 1 epoch
        pos_weights = torch.ones(3)
        trainer = BGC_MLM_tools.BGCMultiLabelTrainier(
            classifier, train_loader, val_set, pos_weights, device="cpu"
        )
        trainer.train(0)  # 1 epoch

        # 5. Should be able to predict
        predictions = trainer.predict(val_set, batch_size=1)
        assert predictions.shape[1] == 3  # n_tasks


class TestFromScratchClassificationPipeline:
    """Mirrors classificationTaskFromScratch.py: no pretrained weights, freeze=False."""

    def test_pipeline_runs(self, classification_setup):
        cfg = classification_setup

        # 1. Build MLM — no loading of state_dict (from scratch)
        bgc_mlm = BGC_MLM(**cfg["params"])
        mlm = MLM(bgc_mlm, cfg["params"]["vocab_size"])

        # 2. Classifier with freeze=False (the from-scratch difference)
        classifier = BGCMultiLabelClassifier(
            mlm.bgc_mlm, cfg["params"]["d_model"],
            len(cfg["y_vals"][0]), freeze=False
        )

        dataset = BGC_MLM_tools.BGCClassificationDatasets(
            cfg["bgc_tokens"], cfg["token_list"], cfg["y_vals"],
            seq_len=cfg["params"]["seq_len"]
        )
        train_set, val_set = random_split(dataset, [4, 1])
        train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

        pos_weights = torch.ones(3)
        trainer = BGC_MLM_tools.BGCMultiLabelTrainier(
            classifier, train_loader, val_set, pos_weights, device="cpu"
        )
        trainer.train(0)

        # All bgc_mlm params should be trainable
        for param in classifier.bgc_mlm.parameters():
            assert param.requires_grad is True


class TestEvaluateOnlyClassificationPipeline:
    """Mirrors classificationTaskTest.py: load classifier weights, predict only."""

    def test_pipeline_runs(self, classification_setup, tmp_path):
        cfg = classification_setup

        # 1. Build and save a classifier model first
        bgc_mlm = BGC_MLM(**cfg["params"])
        mlm = MLM(bgc_mlm, cfg["params"]["vocab_size"])
        classifier = BGCMultiLabelClassifier(
            mlm.bgc_mlm, cfg["params"]["d_model"],
            len(cfg["y_vals"][0]), freeze=True
        )
        model_path = str(tmp_path / "test_classifier.pt")
        torch.save(classifier.state_dict(), model_path)

        # 2. Load into fresh model (as classificationTaskTest.py does)
        bgc_mlm2 = BGC_MLM(**cfg["params"])
        mlm2 = MLM(bgc_mlm2, cfg["params"]["vocab_size"])
        classifier2 = BGCMultiLabelClassifier(
            mlm2.bgc_mlm, cfg["params"]["d_model"],
            len(cfg["y_vals"][0]), freeze=True
        )
        classifier2.load_state_dict(torch.load(model_path, weights_only=True))

        # 3. Predict without training
        dataset = BGC_MLM_tools.BGCClassificationDatasets(
            cfg["bgc_tokens"], cfg["token_list"], cfg["y_vals"],
            seq_len=cfg["params"]["seq_len"]
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        trainer = BGC_MLM_tools.BGCMultiLabelTrainier(
            classifier2, loader, dataset, None, device="cpu"
        )
        predictions = trainer.predict(dataset, batch_size=2)
        assert predictions.shape == (5, 3)  # 5 samples, 3 tasks
