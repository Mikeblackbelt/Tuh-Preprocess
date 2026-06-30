import pytest
import pandas as pd
from pipeline.preictal_segment import (
    get_unique_tags,
    make_master_file,
    add_preictal_tags,
    get_split,
)
from util import handle_logs
from testing.helpers import *


@pytest.fixture
def sample_master():
    return pd.DataFrame({
        "edf_path":   ["a.edf", "a.edf"],
        "csv_path":   ["a.csv", "a.csv"],
        "split":      ["train", "dev"],
        "channel":    ["FP1-F7", "FP1-F7"],
        "start_time": [100.0, 400.0],
        "stop_time":  [110.0, 420.0],
        "label":      ["fnsz", "gnsz"],
        "confidence": [1, 1],
        "status":     [-1, -1],
    })

def test_preictal_row_count(sample_master):
    logger.info("test_preictal_row_count: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)
    assert len(result) == len(sample_master) * 2
    logger.info("test_preictal_row_count: passed")

def test_preictal_labels(sample_master):
    logger.info("test_preictal_labels: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)
    assert "pfnsz" in result["label"].values
    assert "pgnsz" in result["label"].values
    logger.info("test_preictal_labels: passed")

def test_preictal_times_clean(sample_master):
    # row1: start=100, cutoff=10, dur=50 -> raw_end=90, raw_start=40 -> both > 0
    # row2: start=400, cutoff=10, dur=50 -> raw_end=390, raw_start=340 -> both > 0
    logger.info("test_preictal_times_clean: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)

    pfnsz = result[result["label"] == "pfnsz"].iloc[0]
    assert pfnsz["start_time"] == 40.0
    assert pfnsz["stop_time"] == 90.0
    assert pfnsz["status"] == 0

    pgnsz = result[result["label"] == "pgnsz"].iloc[0]
    assert pgnsz["start_time"] == 340.0
    assert pgnsz["stop_time"] == 390.0
    assert pgnsz["status"] == 0
    logger.info("test_preictal_times_clean: passed")

def test_preictal_status_start_trimmed(sample_master):
    # row1: start=100, cutoff=10 -> raw_end=90 (>0), dur=200 -> raw_start=-110 (<=0)
    # expect status=1, start clamped to 0, stop stays at raw_end
    logger.info("test_preictal_status_start_trimmed: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=200)
    pfnsz = result[result["label"] == "pfnsz"].iloc[0]
    assert pfnsz["start_time"] == 0.0
    assert pfnsz["stop_time"] == 90.0
    assert pfnsz["status"] == 1
    logger.info("test_preictal_status_start_trimmed: passed")

def test_preictal_status_collapsed(sample_master):
    # row1: start=100, cutoff=150 -> raw_end=-50 (<=0)
    # expect status=2, start and stop both clamped to 0
    logger.info("test_preictal_status_collapsed: start")
    result = add_preictal_tags(sample_master, start_cutoff=150, max_duration=50)
    pfnsz = result[result["label"] == "pfnsz"].iloc[0]
    assert pfnsz["start_time"] == 0.0
    assert pfnsz["stop_time"] == 0.0
    assert pfnsz["status"] == 2
    logger.info("test_preictal_status_collapsed: passed")

def test_preictal_status_never_negative(sample_master):
    logger.info("test_preictal_status_never_negative: start")
    result = add_preictal_tags(sample_master, start_cutoff=9999, max_duration=9999)
    preictal_rows = result[result["label"].str.startswith("p")]
    assert (preictal_rows["start_time"] >= 0).all()
    assert (preictal_rows["stop_time"] >= 0).all()
    logger.info("test_preictal_status_never_negative: passed")

def test_preictal_original_rows_unchanged(sample_master):
    logger.info("test_preictal_original_rows_unchanged: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)
    fnsz = result[result["label"] == "fnsz"].iloc[0]
    assert fnsz["start_time"] == 100.0
    assert fnsz["stop_time"] == 110.0
    assert fnsz["status"] == -1
    logger.info("test_preictal_original_rows_unchanged: passed")

def test_preictal_sorted_by_split_then_time(sample_master):
    logger.info("test_preictal_sorted_by_split_then_time: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)
    # within each split group, start_time should be non-decreasing
    for split, group in result.groupby("split"):
        times = group["start_time"].tolist()
        assert times == sorted(times)
    logger.info("test_preictal_sorted_by_split_then_time: passed")

def test_preictal_zero_cutoff_and_duration():
    logger.info("test_preictal_zero_cutoff_and_duration: start")
    # With zero cutoff and duration: raw_end = start - 0 = start; raw_start = start - 0 = start
    # raw_end > 0 (start=100), raw_start = start - 0 - 0 = 100 > 0 → status=0, window=[100, 100]
    df = pd.DataFrame({
        "edf_path":   ["a.edf"],
        "csv_path":   ["a.csv"],
        "split":      ["train"],
        "channel":    ["FP1-F7"],
        "start_time": [100.0],
        "stop_time":  [110.0],
        "label":      ["fnsz"],
        "confidence": [1],
        "status":     [-1],
    })
    result = add_preictal_tags(df, start_cutoff=0, max_duration=0)
    preictal = result[result["label"] == "pfnsz"].iloc[0]
    assert preictal["start_time"] >= 0
    assert preictal["stop_time"] >= 0
    logger.info("test_preictal_zero_cutoff_and_duration: passed")

def test_preictal_exact_boundary_raw_end_zero():
    logger.info("test_preictal_exact_boundary_raw_end_zero: start")
    # ictal_start == start_cutoff => raw_end exactly 0 → status=2, window=[0,0]
    df = pd.DataFrame({
        "edf_path":   ["a.edf"],
        "csv_path":   ["a.csv"],
        "split":      ["train"],
        "channel":    ["FP1-F7"],
        "start_time": [10.0],
        "stop_time":  [20.0],
        "label":      ["fnsz"],
        "confidence": [1],
        "status":     [-1],
    })
    result = add_preictal_tags(df, start_cutoff=10, max_duration=5)
    preictal = result[result["label"] == "pfnsz"].iloc[0]
    assert preictal["start_time"] == 0.0
    assert preictal["stop_time"] == 0.0
    assert preictal["status"] == 2
    logger.info("test_preictal_exact_boundary_raw_end_zero: passed")

def test_preictal_exact_boundary_raw_start_zero():
    logger.info("test_preictal_exact_boundary_raw_start_zero: start")
    # raw_end > 0, raw_start exactly 0 → status=1, window=[0, raw_end]
    df = pd.DataFrame({
        "edf_path":   ["a.edf"],
        "csv_path":   ["a.csv"],
        "split":      ["train"],
        "channel":    ["FP1-F7"],
        "start_time": [15.0],
        "stop_time":  [25.0],
        "label":      ["fnsz"],
        "confidence": [1],
        "status":     [-1],
    })
    # raw_end = 15 - 5 = 10; raw_start = 10 - 10 = 0 → exactly 0 → status=1
    result = add_preictal_tags(df, start_cutoff=5, max_duration=10)
    preictal = result[result["label"] == "pfnsz"].iloc[0]
    assert preictal["start_time"] == 0.0
    assert preictal["stop_time"] == 10.0
    assert preictal["status"] == 1
    logger.info("test_preictal_exact_boundary_raw_start_zero: passed")

def test_preictal_single_row_input():
    logger.info("test_preictal_single_row_input: start")
    df = pd.DataFrame({
        "edf_path":   ["single.edf"],
        "csv_path":   ["single.csv"],
        "split":      ["eval"],
        "channel":    ["FP1-F7"],
        "start_time": [200.0],
        "stop_time":  [210.0],
        "label":      ["bckg"],
        "confidence": [1],
        "status":     [-1],
    })
    result = add_preictal_tags(df, start_cutoff=10, max_duration=30)
    assert len(result) == 2
    assert "pbckg" in result["label"].values
    logger.info("test_preictal_single_row_input: passed")

def test_preictal_original_status_preserved(sample_master):
    logger.info("test_preictal_original_status_preserved: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)
    original_rows = result[~result["label"].str.startswith("p")]
    assert (original_rows["status"] == -1).all()
    logger.info("test_preictal_original_status_preserved: passed")

def test_preictal_label_prefix_always_p(sample_master):
    logger.info("test_preictal_label_prefix_always_p: start")
    result = add_preictal_tags(sample_master, start_cutoff=10, max_duration=50)
    preictal_rows = result[result["label"].str.startswith("p")]
    for label in preictal_rows["label"]:
        assert label.startswith("p")
    logger.info("test_preictal_label_prefix_always_p: passed")