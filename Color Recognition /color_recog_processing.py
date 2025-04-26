import os
import glob
import numpy as np
import pandas as pd
import pickle
from scipy.signal import iirnotch, butter, filtfilt

def load_and_crop(csv_file, fs=250, calibration_sec=35):
    """
    Load a CSV file and remove the first calibration period.
    Assumes CSV has either a 'Time' column or only channel columns.
    Returns: data (n_channels, n_samples_after_crop)
    """
    df = pd.read_csv(csv_file)
    data = df.values.T  # shape: (n_channels, n_samples)
    cut_samples = int(calibration_sec * fs)
    return data[:, cut_samples:]

def design_filters(fs=250, notch_freq=60.0, notch_q=30, bp_low=1.0, bp_high=30.0, bp_order=4):
    """
    Design notch and band-pass filters.
    Returns: (b_notch, a_notch), (b_bp, a_bp)
    """
    w0 = notch_freq / (fs / 2)
    b_notch, a_notch = iirnotch(w0, notch_q)
    low = bp_low / (fs / 2)
    high = bp_high / (fs / 2)
    b_bp, a_bp = butter(bp_order, [low, high], btype='band')
    return (b_notch, a_notch), (b_bp, a_bp)

def apply_filters(data, notch_coef, bp_coef):
    """
    Apply notch and band-pass filters to multi-channel data.
    data: (n_channels, n_samples)
    """
    b_notch, a_notch = notch_coef
    b_bp, a_bp       = bp_coef
    filtered = filtfilt(b_notch, a_notch, data, axis=1)
    return filtfilt(b_bp, a_bp, filtered, axis=1)

def extract_epochs(data, fs=250, trials=3, flash_dur=3, inter_flash=3, baseline_dur=10):
    """
    Extract ON (1) and OFF (0) epochs from data.
    Returns: epochs_list, labels_list
    """
    epoch_samps = int(flash_dur * fs)
    trial_samps = int((3*flash_dur + 2*inter_flash + baseline_dur) * fs)
    
    # Offsets for on/off in seconds
    on_offsets  = [0, flash_dur+inter_flash, 2*(flash_dur+inter_flash)]
    off_offsets = [flash_dur, 2*(flash_dur+inter_flash), 3*(flash_dur+inter_flash)]
    
    epochs, labels = [], []
    n_chan, n_samp = data.shape
    
    for t in range(trials):
        base = t * trial_samps
        # ON epochs
        for off in on_offsets:
            start = base + int(off*fs)
            end   = start + epoch_samps
            epochs.append(data[:, start:end]); labels.append(1)
        # OFF epochs (including first 3s of baseline)
        for off in off_offsets:
            start = base + int(off*fs)
            end   = start + epoch_samps
            epochs.append(data[:, start:end]); labels.append(0)
    
    return epochs, labels

def reject_artifacts(epochs, labels, threshold=100.0):
    """
    Remove epochs where any channel exceeds threshold (µV).
    Returns cleaned lists.
    """
    clean_e, clean_l = [], []
    for ep, lbl in zip(epochs, labels):
        if np.any(np.abs(ep) > threshold):
            continue
        clean_e.append(ep); clean_l.append(lbl)
    return clean_e, clean_l

def baseline_correct(epochs, fs=250, baseline_ms=100):
    """
    Subtract mean of first 100 ms from each epoch.
    """
    base_samps = int((baseline_ms/1000)*fs)
    return [ep - ep[:, :base_samps].mean(axis=1, keepdims=True) for ep in epochs]

def preprocess_all(raw_folder, fs=250):
    """
    Full pipeline: load, filter, epoch, reject artifacts, baseline-correct.
    Returns:
      X: (n_epochs, n_channels*750)
      y: (n_epochs,)
    """
    notch_coef, bp_coef = design_filters(fs=fs)
    files = sorted(glob.glob(os.path.join(raw_folder, "*.csv")))
    
    all_epochs, all_labels = [], []
    for fpath in files:
        data   = load_and_crop(fpath, fs=fs)
        data   = apply_filters(data, notch_coef, bp_coef)
        epochs, labels = extract_epochs(data, fs=fs)
        epochs, labels = reject_artifacts(epochs, labels)
        epochs         = baseline_correct(epochs, fs=fs)
        all_epochs.extend(epochs)
        all_labels.extend(labels)
    
    # Stack & flatten
    all_epochs = np.stack(all_epochs, axis=0)  # (n_epochs, n_chan, 750)
    n_epochs, n_chan, n_samp = all_epochs.shape
    X = all_epochs.reshape(n_epochs, n_chan * n_samp)
    y = np.array(all_labels)
    return X, y

if __name__ == "__main__":
    raw_folder = "/Users/adhityaram/Projects/eeg-wheelchair/Color Recognition /color_recog"
    X, y       = preprocess_all(raw_folder)
    print("X shape:", X.shape)  
    print("y shape:", y.shape)

    # Save to pickle file
    output_path = os.path.join(raw_folder, "processed_data.pkl")
    with open(output_path, "wb") as pf:
        pickle.dump({'X': X, 'y': y}, pf)
    print(f"Saved processed data to {output_path}")
