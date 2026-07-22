import mne
import pandas as pd
from util import handle_logs
from testing.helpers import *

logger = handle_logs.get_logger("process_signal", "logs/app.log")

def split_into_epochs(edf_path, epoch_duration=1):
    """
    Load an EDF recording and divide it into consecutive fixed-duration epochs.
    
    Parameters:
        edf_path: Path to the EDF file.
        epoch_duration: Duration of each epoch in seconds.
    
    Returns:
        epochs: MNE Epochs object containing the segmented recording.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True)
    
    events = mne.make_fixed_length_events(
        raw, 
        id=1,
        duration=epoch_duration,
        overlap=0  
    )
    
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=1,
        tmin=0,
        tmax=epoch_duration,
        baseline=None,
        preload=True,
        verbose=False
    )
    return epochs

# These are the standard bipolar pairs in the 10-20 system
default_bipolar_pairs = [
    ('FP1', 'F7'),
    ('F7', 'T3'),
    ('T3', 'T5'),
    ('T5', 'O1'),
    ('FP2', 'F8'),
    ('F8', 'T4'),
    ('T4', 'T6'),
    ('T6', 'O2'),
    ('FZ', 'CZ'),
    ('CZ', 'PZ'),
    ('FP1', 'F3'),
    ('F3', 'C3'),
    ('C3', 'P3'),
    ('P3', 'O1'),
    ('FP2', 'F4'),
    ('F4', 'C4'),
    ('C4', 'P4'),
    ('P4', 'O2'),
]

def create_bipolar_montages(raw, metadata, bipolar_pairs=default_bipolar_pairs):
    """
    Convert .npy file to bipolar montage.
    
    Parameters:
        input_npy_path: Path to input .npy file 
        output_npy_path: Path to save bipolar .npy file (optional)
    
    Returns:
        bipolar_data: numpy array with bipolar channels (18, n_samples)
    """
    raw, metadata = _standardize_channels_names(raw, metadata)

    # Create montage row tuple list (indices to subtract)
    montage_rows_tuple_list = []
    for ch1, ch2 in bipolar_pairs:
        idx1 = channel_index_dict[ch1]
        idx2 = channel_index_dict[ch2]
        montage_rows_tuple_list.append((idx1, idx2))

        difference = ch1_signal - ch2_signal
    
    # Continue accordingly to whether the .edf files are converted to .npy or stay as .edf

