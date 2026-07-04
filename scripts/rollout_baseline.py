from pathlib import Path
import time
import argparse

import gymnasium as gym
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.06)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "baseline" / "ppo_reacher_baseline_final.zip"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = gym.make("Reacher-v5", render_mode="human")

    model = PPO.load(
        str(model_path),
        env=env,
        device=args.device,
    )

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=100 + episode)
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1
            done = terminated or truncated

            time.sleep(args.delay)

        print(
            f"Episode {episode + 1}: "
            f"steps = {step_count}, "
            f"total reward = {total_reward:.3f}"
        )

    input("Press Enter to close the MuJoCo window...")
    env.close()


if __name__ == "__main__":
    main()