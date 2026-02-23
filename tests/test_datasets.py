# -*- coding: utf-8 -*-
"""
Tests for dataset classes (BGCClassificationDatasets, BGCRegressionDatasets,
BGCTargetEmbeddingDataset) from BGC_MLM_tools.py.

These dataset classes are still used by the refactored scripts. Tests verify
that the output dicts, token indexing, and padding match old-script expectations.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'BGC-MLM-old'))
import BGC_MLM_tools


class TestBGCClassificationDatasets:

    def _make_dataset(self, token_list, sample_bgc_tokens):
        y_vals = [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
        ]
        return BGC_MLM_tools.BGCClassificationDatasets(
            sample_bgc_tokens, token_list, y_vals, seq_len=12
        )

    def test_output_keys(self, token_list, sample_bgc_tokens):
        """__getitem__ returns dict with bert_input, bert_label, classification_label."""
        ds = self._make_dataset(token_list, sample_bgc_tokens)
        item = ds[0]
        assert "bert_input" in item
        assert "bert_label" in item
        assert "classification_label" in item

    def test_length(self, token_list, sample_bgc_tokens):
        """__len__ matches number of input sequences."""
        ds = self._make_dataset(token_list, sample_bgc_tokens)
        assert len(ds) == 5

    def test_token_indexing(self, token_list, sample_bgc_tokens):
        """Token strings are converted to their index in token_list."""
        ds = self._make_dataset(token_list, sample_bgc_tokens)
        item = ds[0]
        bert_input = item["bert_input"].tolist()
        # First token is CLS -> index 2
        assert bert_input[0] == token_list.index("CLS")
        # Last non-PAD token before padding is SEP -> index 3
        # seq[0] = [CLS, PF00001, PF00002, PF00003, SEP, PAD, PAD, ...]
        assert bert_input[4] == token_list.index("SEP")

    def test_padding_tokens(self, token_list, sample_bgc_tokens):
        """PAD tokens map to token_list.index('PAD') = 0."""
        ds = self._make_dataset(token_list, sample_bgc_tokens)
        item = ds[0]
        bert_input = item["bert_input"].tolist()
        pad_idx = token_list.index("PAD")
        # Positions after SEP should be PAD
        for i in range(5, 12):
            assert bert_input[i] == pad_idx

    def test_classification_label_values(self, token_list, sample_bgc_tokens):
        """Classification labels match the input y_vals."""
        ds = self._make_dataset(token_list, sample_bgc_tokens)
        item0 = ds[0]
        assert item0["classification_label"].tolist() == [1, 0, 0]
        item2 = ds[2]
        assert item2["classification_label"].tolist() == [1, 1, 0]


class TestBGCRegressionDatasets:

    def _make_dataset(self, token_list, sample_bgc_tokens):
        y_vals = [
            [1.5, 2.3],
            [0.7, 3.1],
            [2.0, 1.0],
            [0.3, 4.5],
            [1.1, 2.8],
        ]
        return BGC_MLM_tools.BGCRegressionDatasets(
            sample_bgc_tokens, token_list, y_vals, seq_len=12
        )

    def test_output_keys(self, token_list, sample_bgc_tokens):
        """__getitem__ returns dict with bert_input, bert_label, regression_value."""
        ds = self._make_dataset(token_list, sample_bgc_tokens)
        item = ds[0]
        assert "bert_input" in item
        assert "bert_label" in item
        assert "regression_value" in item

    def test_length(self, token_list, sample_bgc_tokens):
        ds = self._make_dataset(token_list, sample_bgc_tokens)
        assert len(ds) == 5

    def test_float_values_preserved(self, token_list, sample_bgc_tokens):
        """Regression values are preserved as floats in the tensor."""
        ds = self._make_dataset(token_list, sample_bgc_tokens)
        item = ds[0]
        vals = item["regression_value"].tolist()
        assert abs(vals[0] - 1.5) < 1e-5
        assert abs(vals[1] - 2.3) < 1e-5


class TestBGCTargetEmbeddingDataset:

    def _make_dataset(self, token_list, sample_bgc_tokens):
        target_embeddings = [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ]
        return BGC_MLM_tools.BGCTargetEmbeddingDataset(
            sample_bgc_tokens, token_list, target_embeddings, seq_len=12
        )

    def test_output_keys(self, token_list, sample_bgc_tokens):
        """__getitem__ returns dict with bert_input, bert_label, target_embedding."""
        ds = self._make_dataset(token_list, sample_bgc_tokens)
        item = ds[0]
        assert "bert_input" in item
        assert "bert_label" in item
        assert "target_embedding" in item

    def test_length(self, token_list, sample_bgc_tokens):
        ds = self._make_dataset(token_list, sample_bgc_tokens)
        assert len(ds) == 5

    def test_int_values_preserved(self, token_list, sample_bgc_tokens):
        """Fingerprint integer values are preserved."""
        ds = self._make_dataset(token_list, sample_bgc_tokens)
        item = ds[0]
        assert item["target_embedding"].tolist() == [1, 0, 1, 0]
