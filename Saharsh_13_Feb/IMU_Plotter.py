import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

imu_channels = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z"
]

for trial_no in range(1, 10):
    df = pd.read_csv(f"trial_0{trial_no}/imu_wrist_data.csv")

    N = len(df)
    time_sec = np.linspace(0, 6, N)

    out_dir = Path(f"Wrist_IMU_Plots/Trial {trial_no}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for channel in imu_channels:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(time_sec, df[channel], linewidth=1)

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(channel)
        ax.set_title(f"Trial {trial_no} IMU {channel} Data")

        ax.set_xticks(np.arange(0, 7, 1))
        fig.tight_layout()

        fig.savefig(out_dir / f"{channel}.png", dpi=300)
        plt.close(fig)