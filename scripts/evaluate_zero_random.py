import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path


def run_policy(policy_type: str, seeds):
    env = gym.make("Reacher-v5")
    records = []

    for seed in seeds:
        obs, info = env.reset(seed=seed)

        done = False
        total_reward = 0.0
        total_ctrl_cost = 0.0
        total_dist_penalty = 0.0
        step_count = 0
        last_info = {}

        while not done:
            if policy_type == "zero":
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            elif policy_type == "random":
                action = env.action_space.sample()
            else:
                raise ValueError(f"Unknown policy_type: {policy_type}")

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            total_dist_penalty += float(info.get("reward_dist", 0.0))
            total_ctrl_cost += float(info.get("reward_ctrl", 0.0))

            last_info = info
            step_count += 1
            done = terminated or truncated

        final_distance = -float(last_info.get("reward_dist", np.nan))
        final_ctrl_cost = -float(last_info.get("reward_ctrl", np.nan))

        records.append(
            {
                "policy": policy_type,
                "seed": seed,
                "episode_reward": total_reward,
                "total_dist_penalty": total_dist_penalty,
                "total_ctrl_penalty": total_ctrl_cost,
                "final_distance": final_distance,
                "final_ctrl_cost": final_ctrl_cost,
                "steps": step_count,
            }
        )

    env.close()
    return pd.DataFrame(records)


def summarize(df):
    return (
        df.groupby("policy")
        .agg(
            mean_episode_reward=("episode_reward", "mean"),
            std_episode_reward=("episode_reward", "std"),
            mean_final_distance=("final_distance", "mean"),
            std_final_distance=("final_distance", "std"),
            mean_final_ctrl_cost=("final_ctrl_cost", "mean"),
        )
        .reset_index()
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    result_dir = project_root / "results" / "sanity_check"
    result_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(100, 120))

    zero_df = run_policy("zero", seeds)
    random_df = run_policy("random", seeds)

    all_df = pd.concat([zero_df, random_df], ignore_index=True)
    summary_df = summarize(all_df)

    print("\n=== Zero / Random Policy Sanity Check ===")
    print(summary_df.to_string(index=False))

    all_df.to_csv(result_dir / "zero_random_episode_results.csv", index=False)
    summary_df.to_csv(result_dir / "zero_random_summary.csv", index=False)

    print(f"\nSaved results to: {result_dir}")


if __name__ == "__main__":
    main()