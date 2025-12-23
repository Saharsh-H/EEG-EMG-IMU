import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

for trial_no in range(1,6):
    df = pd.read_csv(f"trial_0{trial_no}/emg_data.csv")

    time_col = "timestamp"
    emg_cols = df.columns.drop(time_col)

    N = len(df)
    time_sec = np.linspace(0, 6, N)

    out_dir = Path(f"EMG_Plots/Trial {trial_no}")
    out_dir.mkdir(parents=True, exist_ok=True)

    channel_no = 0
    for emg_channel in emg_cols:
        channel_no += 1
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(time_sec, df[emg_channel], linewidth=1)

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(f"EMG {channel_no}")
        ax.set_title(f"Trial {trial_no} EMG {channel_no} Channel Data")

        ax.set_xticks(np.arange(0, 7, 1))
        fig.tight_layout()

        fig.savefig(out_dir / f"EMG {channel_no}.png", dpi=300)
        plt.close(fig)