import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# IMU channels
imu_channels = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z"
]

# Correct file path
file_path = Path("trial_04") / "imu_wrist_data.csv"

# Read only required columns
df = pd.read_csv(file_path, usecols=imu_channels)

N = len(df)

# Time axis: 0 to 6 seconds
time_sec = np.linspace(0, 6, N)

# Output folder
out_dir = Path("trial_04") / "IMU_Plots_Trial_04"
out_dir.mkdir(exist_ok=True)

for channel in imu_channels:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(time_sec, df[channel].values, linewidth=1)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(channel)
    ax.set_title(f"Trial-04 Time vs {channel}")

    ax.set_xlim(0, 6)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_dir / f"{channel}.png", dpi=300)
    plt.close(fig)

print("Plots saved successfully in:", out_dir)
