# -*- coding: utf-8 -*-
"""
Tests for src/data/loading.py — file parsing functions.

Validates that parse_classification_file, parse_regression_file, and
parse_fp_file produce output matching the inline parsing done in the
original BGC-MLM-old scripts.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data.loading import parse_classification_file, parse_regression_file, parse_fp_file


# ===== Classification Parsing =====

class TestParseClassificationFile:

    def test_with_header(self, tmp_classification_csv):
        """Header row starting with 'bgc_id' should define class names."""
        types, counts, classifications, total = parse_classification_file(
            tmp_classification_csv
        )
        assert types == ["classA", "classB", "classC"]
        assert len(classifications) == 5
        # bgc_001 = [1,0,0], bgc_003 = [1,1,0], bgc_005 = [1,0,1]
        assert classifications["bgc_001"] == [1, 0, 0]
        assert classifications["bgc_003"] == [1, 1, 0]
        assert classifications["bgc_005"] == [1, 0, 1]
        # counts: classA=3 (001,003,005), classB=2 (002,003), classC=1 (004,005 has C=1)
        assert counts["classA"] == 3
        assert counts["classB"] == 2
        assert counts["classC"] == 2
        assert total == sum(counts.values())

    def test_without_header(self, tmp_classification_csv_no_header):
        """When no 'bgc_id' header, types should be numeric indices."""
        types, counts, classifications, total = parse_classification_file(
            tmp_classification_csv_no_header
        )
        # Should fall back to numeric indices: 0, 1, 2
        assert len(types) == 3
        for t in types:
            assert isinstance(t, int)
        assert len(classifications) == 3

    def test_duplicates_skipped(self, tmp_classification_csv_with_duplicates):
        """Duplicate BGC names should be skipped (first entry kept)."""
        types, counts, classifications, total = parse_classification_file(
            tmp_classification_csv_with_duplicates
        )
        # bgc_002 appears twice, second should be ignored
        assert classifications["bgc_002"] == [0, 1]
        # Total unique BGCs: bgc_001, bgc_002, bgc_003
        assert len(classifications) == 3


# ===== Regression Parsing =====

class TestParseRegressionFile:

    def test_basic_parsing(self, tmp_regression_csv):
        """Parses header + float values correctly."""
        types, values = parse_regression_file(tmp_regression_csv)
        assert types == ["targetA", "targetB"]
        assert len(values) == 5
        assert values["bgc_001"] == [1.5, 2.3]
        assert values["bgc_004"] == [0.3, 4.5]

    def test_values_are_floats(self, tmp_regression_csv):
        """All values must be floats (matching old script's float() cast)."""
        _, values = parse_regression_file(tmp_regression_csv)
        for bgc_vals in values.values():
            for v in bgc_vals:
                assert isinstance(v, float)


# ===== Fingerprint Parsing =====

class TestParseFpFile:

    def test_basic_parsing(self, tmp_fp_csv):
        """Parses header + int fingerprint bits."""
        types, fps = parse_fp_file(tmp_fp_csv)
        assert types == ["bit0", "bit1", "bit2", "bit3"]
        assert len(fps) == 5
        assert fps["bgc_001"] == [1, 0, 1, 0]
        assert fps["bgc_002"] == [0, 1, 0, 1]

    def test_without_header(self, tmp_fp_csv_no_header):
        """Falls back to numeric indices when no 'bgc_id' header."""
        types, fps = parse_fp_file(tmp_fp_csv_no_header)
        assert len(types) == 4
        for t in types:
            assert isinstance(t, int)
        assert len(fps) == 2

    def test_values_are_ints(self, tmp_fp_csv):
        """All fingerprint values must be ints."""
        _, fps = parse_fp_file(tmp_fp_csv)
        for bgc_vals in fps.values():
            for v in bgc_vals:
                assert isinstance(v, int)
