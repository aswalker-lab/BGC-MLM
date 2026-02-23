# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for BGC-MLM test suite.
"""

import os
import sys
import math
import pytest
import torch
import tempfile

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Ensure BGC_MLM_tools (old) is importable for dataset classes still used by new scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'BGC-MLM-old'))


# ---------------------------------------------------------------------------
# Token list
# ---------------------------------------------------------------------------

@pytest.fixture
def token_list():
    """Deterministic small token list matching the special-token conventions."""
    return ["PAD", "MASK", "CLS", "SEP", "UNK",
            "PF00001", "PF00002", "PF00003", "PF00004"]


# ---------------------------------------------------------------------------
# Sample BGC tokens (pre-tokenised sequences)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bgc_tokens():
    """Five short tokenised BGC sequences (max_bgc_length=12)."""
    max_len = 12
    seqs = [
        ["CLS", "PF00001", "PF00002", "PF00003", "SEP"],
        ["CLS", "PF00002", "PF00004", "SEP"],
        ["CLS", "PF00001", "PF00001", "PF00003", "PF00004", "SEP"],
        ["CLS", "PF00003", "SEP"],
        ["CLS", "PF00002", "PF00003", "PF00001", "SEP"],
    ]
    # Pad to max_len
    for seq in seqs:
        while len(seq) < max_len:
            seq.append("PAD")
    return seqs


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model_params():
    """Tiny model hyperparameters for fast tests."""
    return dict(
        vocab_size=9,
        seq_len=12,
        d_model=16,
        n_layers=1,
        heads=2,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Temporary CSV files
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_classification_csv(tmp_path):
    """Creates a small classification CSV with header and 5 BGCs × 3 classes."""
    csv_path = tmp_path / "classifications.csv"
    lines = [
        "bgc_id,classA,classB,classC",
        "bgc_001,1,0,0",
        "bgc_002,0,1,0",
        "bgc_003,1,1,0",
        "bgc_004,0,0,1",
        "bgc_005,1,0,1",
    ]
    csv_path.write_text("\n".join(lines) + "\n")
    return str(csv_path)


@pytest.fixture
def tmp_classification_csv_no_header(tmp_path):
    """Classification CSV without 'bgc_id' header line."""
    csv_path = tmp_path / "classifications_noheader.csv"
    lines = [
        "bgc_001,1,0,0",
        "bgc_002,0,1,0",
        "bgc_003,1,1,0",
    ]
    csv_path.write_text("\n".join(lines) + "\n")
    return str(csv_path)


@pytest.fixture
def tmp_classification_csv_with_duplicates(tmp_path):
    """Classification CSV where bgc_002 appears twice."""
    csv_path = tmp_path / "classifications_dup.csv"
    lines = [
        "bgc_id,classA,classB",
        "bgc_001,1,0",
        "bgc_002,0,1",
        "bgc_002,1,1",  # duplicate — should be skipped
        "bgc_003,1,0",
    ]
    csv_path.write_text("\n".join(lines) + "\n")
    return str(csv_path)


@pytest.fixture
def tmp_regression_csv(tmp_path):
    """Regression CSV with header and 5 BGCs × 2 targets."""
    csv_path = tmp_path / "regression.csv"
    lines = [
        "bgc_id,targetA,targetB",
        "bgc_001,1.5,2.3",
        "bgc_002,0.7,3.1",
        "bgc_003,2.0,1.0",
        "bgc_004,0.3,4.5",
        "bgc_005,1.1,2.8",
    ]
    csv_path.write_text("\n".join(lines) + "\n")
    return str(csv_path)


@pytest.fixture
def tmp_fp_csv(tmp_path):
    """Fingerprint CSV with header and 5 BGCs × 4 FP bits."""
    csv_path = tmp_path / "fingerprints.csv"
    lines = [
        "bgc_id,bit0,bit1,bit2,bit3",
        "bgc_001,1,0,1,0",
        "bgc_002,0,1,0,1",
        "bgc_003,1,1,0,0",
        "bgc_004,0,0,1,1",
        "bgc_005,1,0,0,1",
    ]
    csv_path.write_text("\n".join(lines) + "\n")
    return str(csv_path)


@pytest.fixture
def tmp_fp_csv_no_header(tmp_path):
    """FP CSV without 'bgc_id' header."""
    csv_path = tmp_path / "fp_noheader.csv"
    lines = [
        "bgc_001,1,0,1,0",
        "bgc_002,0,1,0,1",
    ]
    csv_path.write_text("\n".join(lines) + "\n")
    return str(csv_path)
