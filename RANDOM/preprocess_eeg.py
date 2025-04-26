import os
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, welch
from sklearn.decomposition import FastICA
import pickle

# ----------------------------
# Configuration: adjust paths as needed
# ----------------------------
EEG_FOLDER = 'EEG_Data'      # Folder containing EEG CSV subfolders
GUI_BASE   = '.'             # Base directory for GUI CSV folders
OUTPUT_DIR = 'preprocessed'  # Directory where pickle will be saved
OUTPUT_PKL = os.path.join(OUTPUT_DIR, 'processed_data_debug.pkl')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Mapping and Focus Directions
# ----------------------------
GUI_FOLDER_MAP = {
    ('04_21_2025', 'P1'): 'Data Collection - P300 2025-04-21 S1 P1',
    ('04_21_2025', 'P2'): 'Data Collection - P300 2025-04-21 S1 P2',
    ('04_23_2025', 'P1'): 'Data Collection - P300 2025-04-23 S1 P1',
    ('04_23_2025', 'P2'): 'Data Collection - P300 2025-04-23 S1 P2',
    ('04_23_2025_S2', 'P1'): 'Data Collection - P300 2025-04-23 S2 P1',
    ('04_23_2025_S2', 'P2'): 'Data Collection - P300 2025-04-23 S2 P2',
}

FOCUS_DIRECTIONS = {
    '04_21_2025': {'P1': ['Forward']*5 + ['Backward']*5 + ['Left']*5 + ['Right']*5,
                  'P2': ['Forward','Backward','Forward','Forward','Backward','Right','Left','Forward','Backward','Right',
                         'Backward','Left','Left','Right','Left','Backward','Forward','Left','Right','Right']},
    '04_23_2025': {'P1': ['Forward']*5 + ['Backward']*5 + ['Left']*5 + ['Right']*5,
                  'P2': ['Backward','Left','Left','Forward','Backward','Left','Right','Forward','Left','Backward',
                         'Left','Forward','Right','Right','Forward','Right','Forward','Backward','Backward','Right']},
    '04_23_2025_S2': {'P1': ['Forward']*5 + ['Backward']*5 + ['Left']*5 + ['Right']*5,
                     'P2': ['Right','Forward','Backward','Right','Left','Left','Forward','Backward','Right','Left',
                            'Right','Forward','Backward','Backward','Left','Forward','Forward','Left','Backward','Right']}
}

# ----------------------------
# Preprocessing parameters
# ----------------------------
SAMPLE_RATE         = 250
CALIBRATION_SAMPLES = 30 * SAMPLE_RATE
EPOCH_OFFSET        = int(0.3 * SAMPLE_RATE)
EPOCH_LENGTH        = int(0.4 * SAMPLE_RATE)
BASELINE_PRE        = int(0.2 * SAMPLE_RATE)
N_CHANNELS          = 8
FILTER_LOW          = 1
FILTER_HIGH         = 15
ARTIFACT_THRESH_UV  = float('inf')  # disable rejection

# ----------------------------
# Helpers: bandpass, ICA, bandpower
# ----------------------------

def bandpass_filter(data, low, high, fs):
    b, a = butter(4, [low/(0.5*fs), high/(0.5*fs)], btype='band')
    return lfilter(b, a, data, axis=0)


def apply_ica(data):
    ica = FastICA(n_components=N_CHANNELS, random_state=0)
    sources = ica.fit_transform(data)
    kurts = np.abs(pd.DataFrame(sources).kurtosis())
    thresh = kurts.quantile(0.95)
    bad = kurts[kurts > thresh].index.tolist()
    sources[:, bad] = 0
    return ica.inverse_transform(sources)


def extract_bandpower(epoch, trial_i=None, epoch_i=None):
    """Compute average band power for each channel and band using correct PSD assignment."""
    feats = []
    for ch in range(epoch.shape[1]):
        # use 64-point segments for PSD
        nperseg = min(64, len(epoch))
        freqs, psd = welch(epoch[:, ch], fs=SAMPLE_RATE, nperseg=nperseg)
        for low, high in [(1,4), (4,8), (8,12), (12,15)]:
            idx = (freqs >= low) & (freqs <= high)
            if not np.any(idx):
                feats.append(0.0)
            else:
                val = psd[idx].mean()
                feats.append(0.0 if np.isnan(val) else val)
    return feats

# ----------------------------
# Preprocess a single trial
# ----------------------------

def preprocess_trial(eeg_data, log_data, focus_dir, trial_i):
    eeg = eeg_data.values[:, :N_CHANNELS]
    eeg = eeg[CALIBRATION_SAMPLES:]
    eeg = bandpass_filter(eeg, FILTER_LOW, FILTER_HIGH, SAMPLE_RATE)
    eeg = apply_ica(eeg)

    flashes = log_data[log_data['Event'].str.contains('Flash')]
    times = flashes['Real Time'].values - log_data['Real Time'].iloc[0]
    events = flashes['Event'].values

    segments, labels, dirs, bp_feats = [], [], [], []
    for idx, (t, evt) in enumerate(zip(times, events), start=1):
        center = int(t * SAMPLE_RATE)
        start = center + EPOCH_OFFSET
        end = start + EPOCH_LENGTH
        base_start = center - BASELINE_PRE
        base_end = center
        if base_start < 0 or end > len(eeg): continue
        epoch = eeg[start:end]
        base = eeg[base_start:base_end]
        epoch = epoch - base.mean(axis=0)
        segments.append(epoch)
        labels.append(1 if evt.split('[')[-1].strip(']') == focus_dir else 0)
        dirs.append(evt)
        bp_feats.append(extract_bandpower(epoch, trial_i, idx))
    return segments, labels, dirs, bp_feats

# ----------------------------
# Batch preprocess all data
# ----------------------------

def batch_preprocess():
    all_trials = []
    for sess in sorted(os.listdir(EEG_FOLDER)):
        sess_path = os.path.join(EEG_FOLDER, sess)
        if not os.path.isdir(sess_path): continue
        for phase in ['P1', 'P2']:
            eeg_files = [f for f in os.listdir(sess_path) if f.endswith('.csv') and phase in f]
            if not eeg_files: continue
            eeg_df = pd.read_csv(os.path.join(sess_path, eeg_files[0]), header=None)
            gui_folder = GUI_FOLDER_MAP.get((sess, phase))
            if not gui_folder: continue
            gui_path = os.path.join(GUI_BASE, gui_folder)
            for i, focus in enumerate(FOCUS_DIRECTIONS.get(sess, {}).get(phase, []), start=1):
                log_file = f'phase{phase[-1]}_trial{i}_{focus}.csv'
                log_path = os.path.join(gui_path, log_file)
                if not os.path.isfile(log_path): continue
                log_df = pd.read_csv(log_path)
                segs, labs, dirs, bp = preprocess_trial(eeg_df, log_df, focus, i)
                if not segs: continue
                all_trials.append({'session': sess,'phase': phase,'trial': i,'focus': focus,'X': np.array(segs),'y': np.array(labs),'flash_dirs': dirs,'bandpower': np.array(bp)})
    print(f"Total trials: {len(all_trials)}")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(all_trials, f)
    print(f"Saved to {OUTPUT_PKL}")

if __name__ == '__main__':
    batch_preprocess()
