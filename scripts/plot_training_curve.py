from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_monitor_file(log_dir: Path) -> Path:
    candidates = list(log_dir.glob("*monitor.csv")) + list(log_dir.glob("*.csv"))

    if not candidates:
        raise FileNotFoundError(f"No monitor csv file found in {log_dir}")

    return candidates[0]


def main():
    project_root = Path(__file__).resolve().parents[1]

    log_dir = project_root / "logs" / "baseline"
    figure_dir = project_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    monitor_file = find_monitor_file(log_dir)

    print(f"Reading monitor file: {monitor_file}")

    # Stable-Baselines3 Monitor files usually have a JSON metadata line at the top.
    df = pd.read_csv(monitor_file, skiprows=1)

    print("Columns:", df.columns.tolist())
    print(df.head())

    # r: episode reward
    # l: episode length
    # t: wall-clock time
    df["episode"] = range(1, len(df) + 1)
    df["reward_smooth"] = df["r"].rolling(window=20, min_periods=1).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["r"], alpha=0.35, label="Episode reward")
    plt.plot(df["episode"], df["reward_smooth"], linewidth=2, label="Smoothed reward (20 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("PPO Baseline Training Curve on Reacher-v5")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = figure_dir / "baseline_reward_curve.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved figure to: {output_path}")

    if "reward_dist" in df.columns and "reward_ctrl" in df.columns:
        df["reward_dist_smooth"] = df["reward_dist"].rolling(window=20, min_periods=1).mean()
        df["reward_ctrl_smooth"] = df["reward_ctrl"].rolling(window=20, min_periods=1).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(df["episode"], df["reward_dist_smooth"], label="Distance reward")
        plt.plot(df["episode"], df["reward_ctrl_smooth"], label="Control reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward component")
        plt.title("Reward Components During PPO Baseline Training")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        component_output_path = figure_dir / "baseline_reward_components.png"
        plt.savefig(component_output_path, dpi=300)
        plt.show()

        print(f"Saved component figure to: {component_output_path}")


if __name__ == "__main__":
    main()