from pathlib import Path
import argparse
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed-start", type=int, default=200)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "hopper_baseline" / "ppo_hopper_baseline_final.zip"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = gym.make("Hopper-v5", render_mode="human")
    model = PPO.load(str(model_path), device=args.device)

    print(f"Loading Hopper policy from: {model_path}")

    for episode in range(args.num_episodes):
        seed = args.seed_start + episode
        obs, info = env.reset(seed=seed)

        done = False
        total_reward = 0.0
        total_forward_reward = 0.0
        total_ctrl_reward = 0.0
        total_survive_reward = 0.0
        action_norm_sum = 0.0
        step_count = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            total_forward_reward += float(info.get("reward_forward", 0.0))
            total_ctrl_reward += float(info.get("reward_ctrl", 0.0))
            total_survive_reward += float(info.get("reward_survive", 0.0))
            action_norm_sum += float(np.linalg.norm(action))

            last_info = info
            step_count += 1
            done = terminated or truncated

            time.sleep(args.delay)

        mean_action_norm = action_norm_sum / step_count
        final_x_position = float(last_info.get("x_position", np.nan))
        final_x_velocity = float(last_info.get("x_velocity", np.nan))

        print(
            f"Episode {episode + 1} | "
            f"seed = {seed} | "
            f"steps = {step_count} | "
            f"reward = {total_reward:.3f} | "
            f"forward_reward = {total_forward_reward:.3f} | "
            f"survive_reward = {total_survive_reward:.3f} | "
            f"ctrl_reward = {total_ctrl_reward:.3f} | "
            f"final_x = {final_x_position:.3f} | "
            f"final_x_velocity = {final_x_velocity:.3f} | "
            f"mean_action_norm = {mean_action_norm:.4f}"
        )

    input("Press Enter to close the MuJoCo window...")
    env.close()


if __name__ == "__main__":
    main()