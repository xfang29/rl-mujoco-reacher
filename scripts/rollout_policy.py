from pathlib import Path
import argparse
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def get_model_path(project_root: Path, policy_name: str) -> Path:
    if policy_name == "baseline":
        return project_root / "models" / "baseline" / "ppo_reacher_baseline_final.zip"

    if policy_name == "modified":
        return (
            project_root
            / "models"
            / "modified_smooth_0.05"
            / "ppo_reacher_modified_final.zip"
        )

    raise ValueError(f"Unknown policy name: {policy_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=str,
        choices=["baseline", "modified"],
        default="modified",
        help="Choose which trained policy to visualize.",
    )
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.04)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-start", type=int, default=100)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = get_model_path(project_root, args.policy)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading {args.policy} policy from:")
    print(model_path)

    env = gym.make("Reacher-v5", render_mode="human")
    model = PPO.load(str(model_path), device=args.device)

    for episode in range(args.num_episodes):
        seed = args.seed_start + episode
        obs, info = env.reset(seed=seed)

        done = False
        total_reward = 0.0
        step_count = 0
        prev_action = None
        action_delta_sum = 0.0
        action_delta_count = 0
        action_norm_sum = 0.0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            action_norm_sum += float(np.linalg.norm(action))

            if prev_action is not None:
                action_delta_sum += float(np.linalg.norm(action - prev_action))
                action_delta_count += 1

            prev_action = action.copy()
            last_info = info
            step_count += 1
            done = terminated or truncated

            time.sleep(args.delay)

        mean_action_delta = (
            action_delta_sum / action_delta_count if action_delta_count > 0 else 0.0
        )
        mean_action_norm = action_norm_sum / step_count
        final_distance = -float(last_info.get("reward_dist", np.nan))
        final_ctrl_cost = -float(last_info.get("reward_ctrl", np.nan))

        print(
            f"Episode {episode + 1} | "
            f"seed = {seed} | "
            f"steps = {step_count} | "
            f"reward = {total_reward:.3f} | "
            f"final distance = {final_distance:.4f} | "
            f"mean action delta = {mean_action_delta:.5f} | "
            f"mean action norm = {mean_action_norm:.5f} | "
            f"final ctrl cost = {final_ctrl_cost:.5f}"
        )

    input("Press Enter to close the MuJoCo window...")
    env.close()


if __name__ == "__main__":
    main()