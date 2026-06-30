import pytest
from unittest.mock import patch
from util import handle_logs
from util.verify_data import (
    verify_data_path,
    verify_data_integrity,
    list_files_glob,
    validate_input,
)

logger = handle_logs.get_logger("test_verify_data", "logs/test.log")


# --- verify_data_path ---

def test_verify_data_path_existing(tmp_path):
    logger.info("test_verify_data_path_existing: start")
    assert verify_data_path(str(tmp_path)) is True
    logger.info("test_verify_data_path_existing: passed")


def test_verify_data_path_nonexistent(tmp_path):
    logger.info("test_verify_data_path_nonexistent: start")
    missing = str(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="Data path does not exist"):
        verify_data_path(missing)
    logger.info("test_verify_data_path_nonexistent: passed")


def test_verify_data_path_existing_file(tmp_path):
    logger.info("test_verify_data_path_existing_file: start")
    f = tmp_path / "file.txt"
    f.write_text("data")
    assert verify_data_path(str(f)) is True
    logger.info("test_verify_data_path_existing_file: passed")


def test_verify_data_path_returns_true(tmp_path):
    logger.info("test_verify_data_path_returns_true: start")
    result = verify_data_path(str(tmp_path))
    assert result is True
    logger.info("test_verify_data_path_returns_true: passed")


# --- verify_data_integrity ---

def test_verify_data_integrity_valid_dir(tmp_path):
    logger.info("test_verify_data_integrity_valid_dir: start")
    (tmp_path / "data.csv").write_text("col\nval")
    assert verify_data_integrity(str(tmp_path)) is True
    logger.info("test_verify_data_integrity_valid_dir: passed")


def test_verify_data_integrity_not_a_directory(tmp_path):
    logger.info("test_verify_data_integrity_not_a_directory: start")
    f = tmp_path / "file.txt"
    f.write_text("data")
    with pytest.raises(ValueError, match="Path is not a directory"):
        verify_data_integrity(str(f))
    logger.info("test_verify_data_integrity_not_a_directory: passed")


def test_verify_data_integrity_empty_directory(tmp_path):
    logger.info("test_verify_data_integrity_empty_directory: start")
    with pytest.raises(ValueError, match="No files found in directory"):
        verify_data_integrity(str(tmp_path))
    logger.info("test_verify_data_integrity_empty_directory: passed")


def test_verify_data_integrity_multiple_files(tmp_path):
    logger.info("test_verify_data_integrity_multiple_files: start")
    (tmp_path / "a.csv").write_text("col\nval")
    (tmp_path / "b.edf").write_text("edf data")
    assert verify_data_integrity(str(tmp_path)) is True
    logger.info("test_verify_data_integrity_multiple_files: passed")


def test_verify_data_integrity_returns_true(tmp_path):
    logger.info("test_verify_data_integrity_returns_true: start")
    (tmp_path / "file.txt").write_text("data")
    result = verify_data_integrity(str(tmp_path))
    assert result is True
    logger.info("test_verify_data_integrity_returns_true: passed")


# --- list_files_glob ---

def test_list_files_glob_prints_files(tmp_path, capsys):
    logger.info("test_list_files_glob_prints_files: start")
    (tmp_path / "a.csv").write_text("col\nval")
    (tmp_path / "b.edf").write_text("edf")
    list_files_glob(str(tmp_path))
    captured = capsys.readouterr()
    assert "a.csv" in captured.out
    assert "b.edf" in captured.out
    logger.info("test_list_files_glob_prints_files: passed")


def test_list_files_glob_empty_dir_prints_nothing(tmp_path, capsys):
    logger.info("test_list_files_glob_empty_dir_prints_nothing: start")
    list_files_glob(str(tmp_path))
    captured = capsys.readouterr()
    assert captured.out == ""
    logger.info("test_list_files_glob_empty_dir_prints_nothing: passed")


def test_list_files_glob_single_file(tmp_path, capsys):
    logger.info("test_list_files_glob_single_file: start")
    (tmp_path / "rec.csv").write_text("col\nval")
    list_files_glob(str(tmp_path))
    captured = capsys.readouterr()
    assert "rec.csv" in captured.out
    logger.info("test_list_files_glob_single_file: passed")


# --- validate_input ---

def test_validate_input_user_confirms_yes(tmp_path):
    logger.info("test_validate_input_user_confirms_yes: start")
    (tmp_path / "file.csv").write_text("col\nval")
    with patch("builtins.input", return_value="y"):
        result, msg = validate_input(str(tmp_path))
    assert result is True
    assert "successful" in msg.lower()
    logger.info("test_validate_input_user_confirms_yes: passed")


def test_validate_input_user_confirms_uppercase_y(tmp_path):
    logger.info("test_validate_input_user_confirms_uppercase_y: start")
    (tmp_path / "file.csv").write_text("col\nval")
    with patch("builtins.input", return_value="Y"):
        result, msg = validate_input(str(tmp_path))
    assert result is True
    logger.info("test_validate_input_user_confirms_uppercase_y: passed")


def test_validate_input_user_rejects(tmp_path):
    logger.info("test_validate_input_user_rejects: start")
    (tmp_path / "file.csv").write_text("col\nval")
    with patch("builtins.input", return_value="n"):
        result, msg = validate_input(str(tmp_path))
    assert result is False
    assert "failed" in msg.lower()
    logger.info("test_validate_input_user_rejects: passed")


def test_validate_input_nonexistent_path(tmp_path):
    logger.info("test_validate_input_nonexistent_path: start")
    missing = str(tmp_path / "ghost_dir")
    result, msg = validate_input(missing)
    assert result is False
    assert "does not exist" in msg
    logger.info("test_validate_input_nonexistent_path: passed")


def test_validate_input_empty_directory(tmp_path):
    logger.info("test_validate_input_empty_directory: start")
    result, msg = validate_input(str(tmp_path))
    assert result is False
    assert "No files found" in msg
    logger.info("test_validate_input_empty_directory: passed")


def test_validate_input_path_is_file_not_dir(tmp_path):
    logger.info("test_validate_input_path_is_file_not_dir: start")
    f = tmp_path / "file.csv"
    f.write_text("col\nval")
    result, msg = validate_input(str(f))
    assert result is False
    assert "not a directory" in msg.lower()
    logger.info("test_validate_input_path_is_file_not_dir: passed")


def test_validate_input_arbitrary_rejection(tmp_path):
    logger.info("test_validate_input_arbitrary_rejection: start")
    (tmp_path / "file.csv").write_text("col\nval")
    with patch("builtins.input", return_value="no"):
        result, msg = validate_input(str(tmp_path))
    assert result is False
    logger.info("test_validate_input_arbitrary_rejection: passed")