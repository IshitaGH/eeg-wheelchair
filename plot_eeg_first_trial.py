import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
EEG_CSV_PATH = 'EEG_Data/04_23_2025/04_23_2025_S1_P1.csv'  # <-- Hardcode your EEG CSV path here
SAMPLE_RATE = 250  # Hz

# Number of samples to skip
skip_initial = 30 * SAMPLE_RATE      # first 30 seconds
skip_after_wait = 2 * SAMPLE_RATE    # wait 2 seconds
window_samples = 10 * SAMPLE_RATE    # 10-second window

def main():
    # Load EEG data
    df = pd.read_csv(EEG_CSV_PATH, header=None)
    # Ensure we have at least 8 channels
    data = df.iloc[:, :8].values

    # Compute start and end indices for the window
    start_idx = skip_initial + skip_after_wait
    end_idx   = start_idx + window_samples

    # Extract the segment
    segment = data[start_idx:end_idx, :]  # shape (2500, 8)
    time = np.arange(window_samples) / SAMPLE_RATE

    # Plot each channel
    fig, axes = plt.subplots(8, 1, figsize=(10, 12), sharex=True)
    for ch in range(8):
        axes[ch].plot(time, segment[:, ch])
        axes[ch].set_ylabel(f'Ch {ch+1}')
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('EEG Channels 1-8 (10s window)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot average across channels
    avg_signal = segment.mean(axis=1)
    plt.figure(figsize=(10, 3))
    plt.plot(time, avg_signal, color='k')
    plt.title('Average EEG Signal (Channels 1-8)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
