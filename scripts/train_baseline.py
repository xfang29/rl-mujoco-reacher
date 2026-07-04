import argparse
from pathlib import Path

import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy


def make_reacher_env(log_dir: Path, seed: int = 42):
    env = gym.make("Reacher-v5")
    env.reset(seed=seed)

    env = Monitor(
        env,
        filename=str(log_dir / "monitor.csv"),
        info_keywords=("reward_dist", "reward_ctrl"),
    )

    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    log_dir = project_root / "logs" / "baseline"
    model_dir = project_root / "models" / "baseline"
    result_dir = project_root / "results" / "baseline"

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=== Training PPO Baseline on Reacher-v5 ===")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Seed: {args.seed}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    env = make_reacher_env(log_dir, seed=args.seed)

    eval_env = gym.make("Reacher-v5")
    eval_env.reset(seed=args.seed + 1)
    eval_env = Monitor(eval_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=str(model_dir),
        name_prefix="ppo_reacher_checkpoint",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best_model"),
        log_path=str(result_dir),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(project_root / "logs" / "tensorboard"),
        seed=args.seed,
        device=args.device,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="ppo_reacher_baseline",
    )

    final_model_path = model_dir / "ppo_reacher_baseline_final"
    model.save(str(final_model_path))

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )

    print("\n=== Final Evaluation ===")
    print(f"Mean reward over 20 episodes: {mean_reward:.3f}")
    print(f"Std reward over 20 episodes: {std_reward:.3f}")
    print(f"Final model saved to: {final_model_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()