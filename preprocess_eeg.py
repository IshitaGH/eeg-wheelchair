import os
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import pickle

# ----------------------------
# Configuration: adjust paths as needed
# ----------------------------
EEG_FOLDER = 'EEG_Data'      # Folder containing EEG CSV subfolders
GUI_BASE   = '.'             # Base directory for GUI CSV folders
OUTPUT_DIR = 'preprocessed'  # Directory where pickle will be saved
OUTPUT_PKL  = os.path.join(OUTPUT_DIR, 'processed_data_v3.pkl')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Mapping and Focus Directions
GUI_FOLDER_MAP = {
    ('04_21_2025', 'P1'): 'Data Collection - P300 2025-04-21 S1 P1',
    ('04_21_2025', 'P2'): 'Data Collection - P300 2025-04-21 S1 P2',
    ('04_23_2025', 'P1'): 'Data Collection - P300 2025-04-23 S1 P1',
    ('04_23_2025', 'P2'): 'Data Collection - P300 2025-04-23 S1 P2',
    ('04_23_2025_S2', 'P1'): 'Data Collection - P300 2025-04-23 S2 P1',
    ('04_23_2025_S2', 'P2'): 'Data Collection - P300 2025-04-23 S2 P2',
}

FOCUS_DIRECTIONS = {
    '04_21_2025': {
        'P1': ['Forward']*5 + ['Backward']*5 + ['Left']*5 + ['Right']*5 ,
        'P2': ['Forward','Backward','Forward','Forward','Backward','Right','Left','Forward','Backward','Right','Backward','Left','Left','Right','Left','Backward','Forward','Left','Right','Right'],
    },
    '04_23_2025': {
        'P1': ['Forward']*5 + ['Backward']*5 + ['Left']*5 + ['Right']*5,
        'P2': ['Backward','Left','Left','Forward','Backward','Left','Right','Forward','Left','Backward','Left','Forward','Right','Right','Forward','Right','Forward','Backward','Backward','Right'],
    },
    '04_23_2025_S2': {
        'P1': ['Forward']*5 + ['Backward']*5 + ['Left']*5 + ['Right']*5,
        'P2': ['Right','Forward','Backward','Right','Left','Left','Forward','Backward','Right','Left','Right','Forward','Backward','Backward','Left','Forward','Forward','Left','Backward','Right'],
    }
}

# ----------------------------
# Preprocessing parameters
SAMPLE_RATE         = 250
CALIBRATION_SAMPLES = 30 * SAMPLE_RATE
EPOCH_OFFSET        = int(0.3 * SAMPLE_RATE)
EPOCH_LENGTH        = int(0.4 * SAMPLE_RATE)
BASELINE_PRE_SAMPLES = int(0.2 * SAMPLE_RATE)  # 200ms before flash
N_CHANNELS          = 8
FILTER_LOW          = 1
FILTER_HIGH         = 15

# ----------------------------
# Bandpass Filter
# ----------------------------
def bandpass_filter(data, low, high, fs):
    b, a = butter(4, [low/(0.5*fs), high/(0.5*fs)], btype='band')
    return lfilter(b, a, data, axis=0)

# ----------------------------
# Preprocess a single trial with baseline correction
# ----------------------------
def preprocess_trial(eeg_data, log_data, focus_dir):
    # use first N_CHANNELS columns as EEG
    eeg = eeg_data.values[:, :N_CHANNELS]
    # discard calibration period
    eeg = eeg[CALIBRATION_SAMPLES:]

    flashes = log_data[log_data['Event'].str.contains('Flash')]
    times   = flashes['Real Time'].values
    events  = flashes['Event'].values
    offset  = log_data['Real Time'].iloc[0]
    times   = times - offset

    segments = []
    for t, evt in zip(times, events):
        flashed = evt.split('[')[-1].strip(']')
        # compute sample indices
        center_sample = int(t * SAMPLE_RATE)
        start = center_sample + EPOCH_OFFSET
        end   = start + EPOCH_LENGTH
        # baseline window just before flash onset
        base_start = center_sample - BASELINE_PRE_SAMPLES
        base_end   = center_sample

        if base_start < 0 or end > len(eeg):
            continue

        # extract epoch
        epoch = eeg[start:end]
        # bandpass filter
        epoch = bandpass_filter(epoch, FILTER_LOW, FILTER_HIGH, SAMPLE_RATE)
        # baseline correction: subtract mean of pre-flash window
        baseline = eeg[base_start:base_end]
        baseline = bandpass_filter(baseline, FILTER_LOW, FILTER_HIGH, SAMPLE_RATE)
        mean_baseline = baseline.mean(axis=0)
        epoch = epoch - mean_baseline

        label = 1 if flashed == focus_dir else 0
        epoch = epoch - epoch.mean(axis=1, keepdims=True)  # CAR (Common Average Reference)
        segments.append({'segment': epoch, 'label': label, 'flash_dir': flashed})
    return segments

# ----------------------------
# Batch preprocess all data with debug prints
# ----------------------------
def batch_preprocess():
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for EEG folder '{EEG_FOLDER}': exists? {os.path.isdir(EEG_FOLDER)}")
    all_trials = []
    for sess in sorted(os.listdir(EEG_FOLDER)):
        sess_path = os.path.join(EEG_FOLDER, sess)
        if not os.path.isdir(sess_path):
            continue
        print(f"Processing session: {sess}")
        for phase in ['P1', 'P2']:
            eeg_files = [f for f in os.listdir(sess_path) if f.endswith('.csv') and phase in f]
            if not eeg_files:
                print(f"  No EEG file for phase {phase}")
                continue
            eeg_path = os.path.join(sess_path, eeg_files[0])
            eeg_data = pd.read_csv(eeg_path, header=None)

            gui_folder = GUI_FOLDER_MAP.get((sess, phase))
            if not gui_folder:
                continue
            gui_path = os.path.join(GUI_BASE, gui_folder)
            if not os.path.isdir(gui_path):
                print(f"  Missing GUI folder {gui_path}")
                continue

            for i in range(1, 21):
                focus = FOCUS_DIRECTIONS.get(sess, {}).get(phase, [None]*20)[i-1]
                log_file = f'phase{phase[-1]}_trial{i}_{focus}.csv'
                log_path = os.path.join(gui_path, log_file)
                if not os.path.isfile(log_path):
                    print(f"    Missing log file: {log_file}")
                    continue
                log_data = pd.read_csv(log_path)
                segs = preprocess_trial(eeg_data, log_data, focus)
                trial = {
                    'session': sess,
                    'phase': phase,
                    'trial': i,
                    'focus': focus,
                    'X': np.array([s['segment'] for s in segs]),
                    'y': np.array([s['label']   for s in segs]),
                    'flash_dirs': [s['flash_dir'] for s in segs]
                }
                all_trials.append(trial)
    print(f"Total trials processed: {len(all_trials)}")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(all_trials, f)
    print(f"Saved {len(all_trials)} trials to {os.path.abspath(OUTPUT_PKL)}")

if __name__ == '__main__':
    print("Starting preprocessing...")
    batch_preprocess()
