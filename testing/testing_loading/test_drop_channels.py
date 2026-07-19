import os
import pytest
import mne
import numpy as np
import pandas as pd
import tempfile
from pipeline.process_signal import standardize_channel_name, load_edf, drop_channels, reorder_channels

from util import handle_logs

logger = handle_logs.get_logger("test_channel_standardize", "logs/test.log")

@pytest.fixture
def test_data(tmp_path):
    """Create two test EDF files: one with extra channels, one missing channels."""
    sfreq = 256
    n_samples = 1000
    
    def create_edf(ch_names, filename):
        n_channels = len(ch_names)
        data = np.random.randn(n_channels, n_samples)
        info = mne.create_info(ch_names, sfreq, ch_types=['eeg'] * n_channels)
        raw = mne.io.RawArray(data, info)
        file_path = tmp_path / filename
        raw.export(str(file_path), overwrite=True)
        return file_path
    
    # Extra channels (PHOTIC, DC1)
    extra = ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F7-LE', 'PHOTIC PH', 'DC1-DC']
    extra_path = create_edf(extra, "extra_channels.edf")
    
    # Missing channels (only 2 channels)
    missing = ['EEG FP2-LE', 'EEG FP1-LE']
    missing_path = create_edf(missing, "missing_channels.edf")
    
    return {
        "extra_path": extra_path,
        "missing_path": missing_path,
    }

desired_order = ['FP1', 'FP2', 'F7', 'F3']

"""The standrdize channel tests are here because the standardize channel function is ran in the drop channels function"""

def test_standardize_channel_name_strips_prefix():
    logger.info("test_standardize_channel_name_strips_prefix_and_suffix: start")
    assert standardize_channel_name('EEG FP1-LE') == 'FP1'
    assert standardize_channel_name('EEG FP2-REF') == 'FP2'
    logger.info("test_standardize_channel_name_strips_prefix_and_suffix: passed")

def test_standardize_channel_name_returns_none_for_non_eeg():
    logger.info("test_standardize_channel_name_returns_none_for_non_eeg: start")
    assert standardize_channel_name('PHOTIC PH') is None
    logger.info("test_standardize_channel_name_returns_none_for_non_eeg: passed")

def test_drop_channels_not_in_desired_order(test_data):
    logger.info("test_drop_channels_not_in_desired_order: start")
    raw, metadata = load_edf(test_data["extra_path"])
    raw, metadata = drop_channels(raw, metadata, desired_order)
    assert sorted(raw.ch_names) == sorted(desired_order)
    assert sorted(metadata['channels'].iloc[0]) == sorted(desired_order)
    logger.info("test_drop_channels_not_in_desired_order: passed")

def test_drop_channels_return_none_on_missing_channels(test_data):
    logger.info("test_drop_channels_return_none_on_missing_channels: start")
    raw, metadata = load_edf(test_data["missing_path"])
    assert drop_channels(raw, metadata, desired_order) is None
    logger.info("test_drop_channels_return_none_on_missing_channels: passed")

def test_drop_channels_raw_and_metadata_channels_match(test_data):
    logger.info("test_drop_channels_raw_and_metadata_channels_match: start")
    raw, metadata = load_edf(test_data["extra_path"])
    raw, metadata = drop_channels(raw, metadata, desired_order)
    assert sorted(raw.ch_names) == sorted(metadata['channels'].iloc[0])
    logger.info("test_drop_channels_raw_and_metadata_channels_match: passed")

def test_reorder_channels(test_data):
    logger.info("test_reorder_channels: start")
    raw, metadata = load_edf(test_data["missing_path"])
    desired_order = ['EEG FP1-LE', 'EEG FP2-LE']
    ordered_raw, ordered_metadata = reorder_channels(raw, metadata, desired_order)

    logger.info('test_reorder_channels, raw: start')
    assert ordered_raw.ch_names == desired_order
    logger.info('test_reorder_channels, raw: passed')

    logger.info('test_reorder_channels, metadata: start')
    assert ordered_metadata['channels'].iloc[0] == desired_order
    logger.info('test_reorder_channels, metadata: passed')

    logger.info("test_reorder_channels: passed")

