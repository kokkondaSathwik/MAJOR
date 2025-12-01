# make_sample_eeg.py
import numpy as np
import pandas as pd

def make_sample(path="sample_eeg.csv", channels=8, samples=512):
    # simulate some EEG-like oscillatory signals
    t = np.linspace(0, 1, samples)
    data = []
    for ch in range(channels):
        freq = 5 + ch  # different freq per channel
        signal = 0.5*np.sin(2*np.pi*freq*t) + 0.05*np.random.randn(samples)
        data.append(signal)
    df = pd.DataFrame(np.array(data).T, columns=[f"ch{i+1}" for i in range(channels)])
    df.to_csv(path, index=False)
    print("Wrote", path)

if __name__ == "__main__":
    make_sample()