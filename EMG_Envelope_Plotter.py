import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

num_trials = 26
base_dir = Path("Saharsh_13_Feb")

for trial_no in range(1, num_trials + 1):
    df = pd.read_csv(base_dir / f"trial_{trial_no:02d}" / "emg_envelope.csv")

    time_col = "timestamp"
    emg_cols = df.columns.drop(time_col)

    N = len(df)
    time_sec = np.linspace(0, 6, N)

    out_dir = base_dir / "Plots/EMG_Envelope_Plots" / f"Trial {trial_no}"
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_no = 0
    for emg_channel in emg_cols:
        channel_no += 1
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(time_sec, df[emg_channel], linewidth=1)

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(f"EMG {channel_no}")
        ax.set_title(f"Trial {trial_no} EMG {channel_no} Channel Envelope")

        ax.set_xticks(np.arange(0, 7, 1))
        fig.tight_layout()

        fig.savefig(out_dir / f"EMG {channel_no} Envelope.png", dpi=300)
        plt.close(fig)