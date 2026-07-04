from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs" / "hopper_baseline"
    figure_dir = project_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    monitor_file = log_dir / "monitor.csv"

    if not monitor_file.exists():
        raise FileNotFoundError(f"Monitor file not found: {monitor_file}")

    df = pd.read_csv(monitor_file, skiprows=1)
    df["episode"] = range(1, len(df) + 1)
    df["reward_smooth"] = df["r"].rolling(window=20, min_periods=1).mean()
    df["length_smooth"] = df["l"].rolling(window=20, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["r"], alpha=0.35, label="Episode reward")
    plt.plot(df["episode"], df["reward_smooth"], linewidth=2, label="Smoothed reward (20 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("PPO Training Curve on Hopper-v5")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    reward_output = figure_dir / "hopper_reward_curve.png"
    plt.savefig(reward_output, dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["l"], alpha=0.35, label="Episode length")
    plt.plot(df["episode"], df["length_smooth"], linewidth=2, label="Smoothed length (20 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Episode length")
    plt.title("Episode Length During PPO Training on Hopper-v5")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    length_output = figure_dir / "hopper_episode_length_curve.png"
    plt.savefig(length_output, dpi=300)
    plt.show()

    print(f"Saved reward curve to: {reward_output}")
    print(f"Saved episode length curve to: {length_output}")


if __name__ == "__main__":
    main()