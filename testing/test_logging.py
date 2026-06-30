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

def test_logging_creates_file(tmp_path):
    logger.info("test_logging_creates_file: start")
    log_file = tmp_path / "test.log"
    test_logger = handle_logs.get_logger("test_logger", str(log_file))
    test_logger.info("This is a test log message.")
    assert log_file.exists(), "Log file was not created"
    logger.info("test_logging_creates_file: passed")

def test_logging_writes_content(tmp_path):
    logger.info("test_logging_writes_content: start")
    log_file = tmp_path / "test.log"
    test_logger = handle_logs.get_logger("test_logger_content", str(log_file))
    test_logger.info("hello from test")
    assert "hello from test" in log_file.read_text()
    logger.info("test_logging_writes_content: passed")

def test_logging_same_name_returns_same_logger(tmp_path):
    logger.info("test_logging_same_name_returns_same_logger: start")
    log_file = tmp_path / "dedup.log"
    l1 = handle_logs.get_logger("dedup_logger", str(log_file))
    l2 = handle_logs.get_logger("dedup_logger", str(log_file))
    assert l1 is l2
    logger.info("test_logging_same_name_returns_same_logger: passed")

def test_logging_no_duplicate_handlers(tmp_path):
    logger.info("test_logging_no_duplicate_handlers: start")
    log_file = tmp_path / "nodup.log"
    l1 = handle_logs.get_logger("nodup_logger", str(log_file))
    initial_handler_count = len(l1.handlers)
    l2 = handle_logs.get_logger("nodup_logger", str(log_file))
    assert len(l2.handlers) == initial_handler_count
    logger.info("test_logging_no_duplicate_handlers: passed")

def test_logging_without_log_file_no_file_created(tmp_path):
    logger.info("test_logging_without_log_file_no_file_created: start")
    handle_logs.get_logger("console_only_logger")
    assert not any(tmp_path.iterdir())
    logger.info("test_logging_without_log_file_no_file_created: passed")

def test_logging_creates_nested_parent_dirs(tmp_path):
    logger.info("test_logging_creates_nested_parent_dirs: start")
    log_file = tmp_path / "deep" / "nested" / "dir" / "app.log"
    test_logger = handle_logs.get_logger("nested_dir_logger", str(log_file))
    test_logger.info("nested log entry")
    assert log_file.exists()
    assert "nested log entry" in log_file.read_text()
    logger.info("test_logging_creates_nested_parent_dirs: passed")

def test_logging_format_includes_level(tmp_path):
    logger.info("test_logging_format_includes_level: start")
    log_file = tmp_path / "format.log"
    test_logger = handle_logs.get_logger("format_logger", str(log_file))
    test_logger.warning("watch out")
    content = log_file.read_text()
    assert "WARNING" in content
    assert "watch out" in content
    logger.info("test_logging_format_includes_level: passed")
