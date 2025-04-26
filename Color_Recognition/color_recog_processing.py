import os
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, butter, filtfilt

def load_and_crop(csv_file, fs=250, calibration_sec=35):
    df = pd.read_csv(csv_file)
    data = df.values.T  # (n_channels, n_samples)
    cut_samples = int(calibration_sec * fs)
    return data[:, cut_samples:]

def design_filters(fs=250, notch_freq=60.0, notch_q=30, bp_low=1.0, bp_high=30.0, bp_order=4):
    w0 = notch_freq / (fs / 2)
    b_notch, a_notch = iirnotch(w0, notch_q)
    low = bp_low / (fs / 2)
    high = bp_high / (fs / 2)
    b_bp, a_bp = butter(bp_order, [low, high], btype='band')
    return (b_notch, a_notch), (b_bp, a_bp)

def apply_filters(data, notch_coef, bp_coef):
    b_notch, a_notch = notch_coef
    b_bp, a_bp       = bp_coef
    filtered = filtfilt(b_notch, a_notch, data, axis=1)
    return filtfilt(b_bp, a_bp, filtered, axis=1)

def extract_epochs(data, fs=250, trials=3, flash_dur=3, inter_flash=3, baseline_dur=10, epoch_ms=300):
    epoch_samps = int(epoch_ms * fs / 1000)  # Convert ms to samples (300ms)

    # The offset times for on and off periods (still keep them based on the old flash durations)
    on_offsets  = [0, flash_dur + inter_flash, 2 * (flash_dur + inter_flash)]
    off_offsets = [flash_dur, 2 * (flash_dur + inter_flash), 3 * (flash_dur + inter_flash)]

    epochs, labels = [], []
    n_chan, n_samp = data.shape

    for t in range(trials):
        base = t * int((3 * flash_dur + 2 * inter_flash + baseline_dur) * fs)
        for off in on_offsets:
            start = base + int(off * fs)
            end = start + epoch_samps
            if end <= n_samp:
                epochs.append(data[:, start:end])
                labels.append(1)
        for off in off_offsets:
            start = base + int(off * fs)
            end = start + epoch_samps
            if end <= n_samp:
                epochs.append(data[:, start:end])
                labels.append(0)

    return epochs, labels

def reject_artifacts(epochs, labels, threshold=100.0):
    clean_e, clean_l = [], []
    for ep, lbl in zip(epochs, labels):
        if np.any(np.abs(ep) > threshold):
            continue
        clean_e.append(ep)
        clean_l.append(lbl)
    return clean_e, clean_l

def baseline_correct(epochs, fs=250, baseline_ms=100):
    base_samps = int((baseline_ms / 1000) * fs)
    return [ep - ep[:, :base_samps].mean(axis=1, keepdims=True) for ep in epochs]

def average_channels(epochs):
    """
    Average across channels for each epoch.
    Input: list of (n_channels, n_samples)
    Output: list of (1, n_samples)
    """
    return [ep.mean(axis=0, keepdims=True) for ep in epochs]

def plot_erp(epochs, labels, fs=250):
    epochs = np.stack(epochs)  # (n_epochs, 1, n_samp)
    labels = np.array(labels)

    on_avg  = epochs[labels == 1].mean(axis=0)  # (1, n_samp)
    off_avg = epochs[labels == 0].mean(axis=0)

    time = np.arange(epochs.shape[2]) / fs * 1000  # ms

    plt.figure(figsize=(10, 5))
    plt.plot(time, on_avg[0], label='ON (stimulus)', color='red')
    plt.plot(time, off_avg[0], label='OFF (baseline)', color='blue')
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Averaged ERP across channels')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def preprocess_all(raw_folder, fs=250, epoch_ms=300):
    notch_coef, bp_coef = design_filters(fs=fs)
    files = sorted(glob.glob(os.path.join(raw_folder, "*.csv")))

    all_epochs, all_labels = [], []
    for fpath in files:
        data   = load_and_crop(fpath, fs=fs)
        data   = apply_filters(data, notch_coef, bp_coef)
        epochs, labels = extract_epochs(data, fs=fs, epoch_ms=epoch_ms)
        epochs, labels = reject_artifacts(epochs, labels)
        epochs         = baseline_correct(epochs, fs=fs)
        epochs         = average_channels(epochs)  # <<<<<< Averaging here
        all_epochs.extend(epochs)
        all_labels.extend(labels)

    all_epochs = np.stack(all_epochs, axis=0)  # (n_epochs, 1, 250)
    n_epochs, n_chan, n_samp = all_epochs.shape
    X = all_epochs.reshape(n_epochs, n_chan * n_samp)
    y = np.array(all_labels)
    return X, y, all_epochs, all_labels

if __name__ == "__main__":
    raw_folder = "/Users/adhityaram/Projects/eeg-wheelchair/Color_Recognition/color_recog"
    X, y, raw_epochs, raw_labels = preprocess_all(raw_folder, epoch_ms=300)  # 300ms epoch size
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Save to pickle
    output_path = os.path.join(raw_folder, "processed_data.pkl")
    with open(output_path, "wb") as pf:
        pickle.dump({'X': X, 'y': y}, pf)
    print(f"Saved processed data to {output_path}")

    # Plot ERP waveforms
    plot_erp(raw_epochs, raw_labels)