from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


def evaluate_model(model_path: Path, seeds, device: str = "cpu"):
    env = gym.make("Reacher-v5")
    model = PPO.load(str(model_path), device=device)

    episode_records = []

    for seed in seeds:
        obs, info = env.reset(seed=seed)

        done = False
        total_reward = 0.0
        total_dist_penalty = 0.0
        total_ctrl_penalty = 0.0
        action_delta_sum = 0.0
        action_delta_count = 0
        action_norm_sum = 0.0
        step_count = 0
        prev_action = None
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            total_dist_penalty += float(info.get("reward_dist", 0.0))
            total_ctrl_penalty += float(info.get("reward_ctrl", 0.0))

            action_norm_sum += float(np.linalg.norm(action))

            if prev_action is not None:
                action_delta_sum += float(np.linalg.norm(action - prev_action))
                action_delta_count += 1

            prev_action = action.copy()
            last_info = info
            step_count += 1
            done = terminated or truncated

        mean_action_delta = (
            action_delta_sum / action_delta_count
            if action_delta_count > 0
            else 0.0
        )

        mean_action_norm = action_norm_sum / step_count
        final_distance = -float(last_info.get("reward_dist", np.nan))
        final_ctrl_cost = -float(last_info.get("reward_ctrl", np.nan))

        episode_records.append(
            {
                "seed": seed,
                "episode_reward": total_reward,
                "total_dist_penalty": total_dist_penalty,
                "total_ctrl_penalty": total_ctrl_penalty,
                "mean_action_delta": mean_action_delta,
                "mean_action_norm": mean_action_norm,
                "final_distance": final_distance,
                "final_ctrl_cost": final_ctrl_cost,
                "steps": step_count,
            }
        )

    env.close()
    return pd.DataFrame(episode_records)


def summarize(df: pd.DataFrame, name: str):
    summary = {
        "policy": name,
        "mean_episode_reward": df["episode_reward"].mean(),
        "std_episode_reward": df["episode_reward"].std(),
        "mean_final_distance": df["final_distance"].mean(),
        "std_final_distance": df["final_distance"].std(),
        "mean_action_delta": df["mean_action_delta"].mean(),
        "std_action_delta": df["mean_action_delta"].std(),
        "mean_action_norm": df["mean_action_norm"].mean(),
        "std_action_norm": df["mean_action_norm"].std(),
    }
    return summary


def plot_training_curves(project_root: Path, figure_dir: Path):
    baseline_monitor = project_root / "logs" / "baseline" / "monitor.csv"
    modified_monitor = project_root / "logs" / "modified_smooth_0.05" / "monitor.csv"

    if not baseline_monitor.exists() or not modified_monitor.exists():
        print("Monitor files not found. Skipping training curve comparison.")
        return

    baseline_df = pd.read_csv(baseline_monitor, skiprows=1)
    modified_df = pd.read_csv(modified_monitor, skiprows=1)

    baseline_df["episode"] = range(1, len(baseline_df) + 1)
    modified_df["episode"] = range(1, len(modified_df) + 1)

    baseline_df["reward_smooth"] = baseline_df["r"].rolling(
        window=20, min_periods=1
    ).mean()

    modified_df["reward_smooth"] = modified_df["r"].rolling(
        window=20, min_periods=1
    ).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(
        baseline_df["episode"],
        baseline_df["reward_smooth"],
        label="Baseline PPO",
        linewidth=2,
    )
    plt.plot(
        modified_df["episode"],
        modified_df["reward_smooth"],
        label="Modified reward PPO",
        linewidth=2,
    )
    plt.xlabel("Episode")
    plt.ylabel("Smoothed episode reward")
    plt.title("Baseline vs Modified Reward Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = figure_dir / "baseline_vs_modified_training_curve.png"
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved training curve comparison to: {output_path}")


def plot_evaluation_comparison(summary_df: pd.DataFrame, figure_dir: Path):
    metrics = [
        "mean_episode_reward",
        "mean_final_distance",
        "mean_action_delta",
        "mean_action_norm",
    ]

    labels = [
        "Episode reward",
        "Final distance",
        "Action delta",
        "Action norm",
    ]

    for metric, label in zip(metrics, labels):
        plt.figure(figsize=(6, 5))
        plt.bar(summary_df["policy"], summary_df[metric])
        plt.ylabel(label)
        plt.title(f"Baseline vs Modified: {label}")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        output_path = figure_dir / f"comparison_{metric}.png"
        plt.savefig(output_path, dpi=300)
        plt.show()

        print(f"Saved figure to: {output_path}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    figure_dir = project_root / "figures"
    result_dir = project_root / "results" / "comparison"

    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    baseline_model = (
        project_root
        / "models"
        / "baseline"
        / "ppo_reacher_baseline_final.zip"
    )

    modified_model = (
        project_root
        / "models"
        / "modified_smooth_0.05"
        / "ppo_reacher_modified_final.zip"
    )

    if not baseline_model.exists():
        raise FileNotFoundError(f"Baseline model not found: {baseline_model}")

    if not modified_model.exists():
        raise FileNotFoundError(f"Modified model not found: {modified_model}")

    seeds = list(range(100, 120))

    print("Evaluating baseline policy...")
    baseline_df = evaluate_model(baseline_model, seeds=seeds, device="cpu")

    print("Evaluating modified reward policy...")
    modified_df = evaluate_model(modified_model, seeds=seeds, device="cpu")

    baseline_df["policy"] = "Baseline"
    modified_df["policy"] = "Modified"

    all_eval_df = pd.concat([baseline_df, modified_df], ignore_index=True)
    all_eval_path = result_dir / "baseline_vs_modified_episode_results.csv"
    all_eval_df.to_csv(all_eval_path, index=False)

    summary_df = pd.DataFrame(
        [
            summarize(baseline_df, "Baseline"),
            summarize(modified_df, "Modified"),
        ]
    )

    summary_path = result_dir / "baseline_vs_modified_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Evaluation Summary on Original Reacher-v5 Reward ===")
    print(summary_df.to_string(index=False))

    print(f"\nSaved episode-level results to: {all_eval_path}")
    print(f"Saved summary results to: {summary_path}")

    plot_training_curves(project_root, figure_dir)
    plot_evaluation_comparison(summary_df, figure_dir)


if __name__ == "__main__":
    main()