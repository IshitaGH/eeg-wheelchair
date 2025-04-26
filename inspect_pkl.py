#!/usr/bin/env python3
import sys
import pickle
import numpy as np

def summarize(obj, indent=0):
    prefix = ' ' * indent
    if isinstance(obj, dict):
        print(f"{prefix}dict with keys: {list(obj.keys())}")
        for k, v in obj.items():
            print(f"{prefix}  └─ {k!r}: ", end='')
            summarize(v, indent+4)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} of length {len(obj)}")
        if len(obj) > 0:
            print(f"{prefix}  └─ First element:")
            summarize(obj[0], indent+4)
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}ndarray, dtype={obj.dtype}, shape={obj.shape}")
    else:
        print(f"{prefix}{type(obj).__name__}: {repr(obj)[:80]}")

def main(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded pickle from: {pkl_path}")
    # Top‐level: list or dict?
    if isinstance(data, list):
        print(f"Top-level: list of length {len(data)}\n")
        sample_n = min(3, len(data))
        for i in range(sample_n):
            print(f"--- Entry {i} ---")
            summarize(data[i], indent=2)
            print()
    else:
        print("Top-level object:")
        summarize(data)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python inspect_pickle.py path/to/file.pkl")
        sys.exit(1)
    main(sys.argv[1])
