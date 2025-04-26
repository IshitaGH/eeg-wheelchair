import os
import pickle
import numpy as np

# Path to your processed pickle file
data_path = os.path.join('preprocessed', 'processed_data_v4.pkl')

# Load the pickle
with open(data_path, 'rb') as f:
    trials = pickle.load(f)

print(f"Loaded pickle from: {os.path.abspath(data_path)}")
print(f"Total number of trials: {len(trials)}")

# Inspect structure of first few trials
for idx, trial in enumerate(trials[:3], start=1):
    print(f"\nTrial {idx} metadata:")
    print(f"  session   : {trial['session']}")
    print(f"  phase     : {trial['phase']}")
    print(f"  trial num : {trial['trial']}")
    print(f"  focus     : {trial['focus']}")
    print(f"  flash_dirs: {trial['flash_dirs'][:5]}...")
    print(f"  X shape   : {trial['X'].shape}  # (n_flashes, 100, 8)")
    print(f"  y shape   : {trial['y'].shape}  # (n_flashes,)\n")

# Overall label distribution
y_all = np.concatenate([t['y'] for t in trials])
zeros = np.sum(y_all == 0)
ones  = np.sum(y_all == 1)
print(f"Overall label distribution: 0 (non-target) = {zeros}, 1 (target) = {ones}")

# Check for any empty trials
empty = [i for i, t in enumerate(trials) if t['X'].size == 0]
print(f"Trials with no segments (should be none): {empty}")

# Example of raw segment values (first channel of first flash of first trial)
if trials and trials[0]['X'].size:
    sample = trials[0]['X'][0, :, 0]
    print(f"First epoch, channel 0 sample (first 10 points): {sample[:10]}")
