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
logger = handle_logs.get_logger("test_pipeline", "logs/test.log")

def test_single_file_single_tag(dataset_dir):
    logger.info("test_single_file_single_tag: start")
    write_csv(dataset_dir / "test.csv", ["bckg"])
    assert get_unique_tags(dataset_dir) == {"bckg"}
    logger.info("test_single_file_single_tag: passed")

def test_single_file_multiple_tags(dataset_dir):
    logger.info("test_single_file_multiple_tags: start")
    write_csv(dataset_dir / "test.csv", ["bckg", "fnsz", "gnsz"])
    assert get_unique_tags(dataset_dir) == {"bckg", "fnsz", "gnsz"}
    logger.info("test_single_file_multiple_tags: passed")

def test_multiple_files_deduped(dataset_dir):
    logger.info("test_multiple_files_deduped: start")
    write_csv(dataset_dir / "a.csv", ["bckg", "fnsz"])
    write_csv(dataset_dir / "b.csv", ["fnsz", "gnsz"])
    assert get_unique_tags(dataset_dir) == {"bckg", "fnsz", "gnsz"}
    logger.info("test_multiple_files_deduped: passed")

def test_nested_directories(dataset_dir):
    logger.info("test_nested_directories: start")
    subdir = dataset_dir / "subdir"
    subdir.mkdir()
    write_csv(dataset_dir / "a.csv", ["bckg"])
    write_csv(subdir / "b.csv", ["fnsz"])
    assert get_unique_tags(dataset_dir) == {"bckg", "fnsz"}
    logger.info("test_nested_directories: passed")

def test_empty_directory(dataset_dir):
    logger.info("test_empty_directory: start")
    assert get_unique_tags(dataset_dir) == set()
    logger.info("test_empty_directory: passed")

def test_ignores_non_csv_files(dataset_dir):
    logger.info("test_ignores_non_csv_files: start")
    (dataset_dir / "test.edf").write_text("not a csv")
    assert get_unique_tags(dataset_dir) == set()
    logger.info("test_ignores_non_csv_files: passed")

def test_malformed_csv_skipped(dataset_dir):
    logger.info("test_malformed_csv_skipped: start")
    (dataset_dir / "bad.csv").write_text("not,valid,csv\n!!!\n")
    write_csv(dataset_dir / "good.csv", ["bckg"])
    assert "bckg" in get_unique_tags(dataset_dir)
    logger.info("test_malformed_csv_skipped: passed")


def test_csv_with_comment_lines_skipped(dataset_dir):
    logger.info("test_csv_with_comment_lines_skipped: start")
    # CSV with '#' comment header lines (like TUSZ annotation format)
    content = "# comment line\nchannel,start_time,stop_time,label,confidence\nFP1-F7,0.0,1.0,fnsz,1\n"
    (dataset_dir / "annotated.csv").write_text(content)
    tags = get_unique_tags(dataset_dir)
    assert "fnsz" in tags
    logger.info("test_csv_with_comment_lines_skipped: passed")

def test_csv_with_whitespace_column_names(dataset_dir):
    logger.info("test_csv_with_whitespace_column_names: start")
    # Column names with surrounding whitespace should still work
    content = " channel , start_time , stop_time , label , confidence \nFP1-F7,0.0,1.0,bckg,1\n"
    (dataset_dir / "whitespace_cols.csv").write_text(content)
    tags = get_unique_tags(dataset_dir)
    assert "bckg" in tags
    logger.info("test_csv_with_whitespace_column_names: passed")

def test_returns_set_type(dataset_dir):
    logger.info("test_returns_set_type: start")
    write_csv(dataset_dir / "test.csv", ["fnsz"])
    result = get_unique_tags(dataset_dir)
    assert isinstance(result, set)
    logger.info("test_returns_set_type: passed")

def test_csv_missing_label_column_skipped(dataset_dir):
    logger.info("test_csv_missing_label_column_skipped: start")
    # CSV without a 'label' column should be skipped without crashing
    (dataset_dir / "no_label.csv").write_text("channel,start_time,stop_time\nFP1-F7,0.0,1.0\n")
    write_csv(dataset_dir / "good.csv", ["fnsz"])
    tags = get_unique_tags(dataset_dir)
    assert "fnsz" in tags
    logger.info("test_csv_missing_label_column_skipped: passed")

# --- get_split ---