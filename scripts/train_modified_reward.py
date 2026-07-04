import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy


class ActionSmoothnessRewardWrapper(gym.Wrapper):
    """
    Add an action smoothness penalty to the original Reacher-v5 reward.

    original_reward = reward_dist + reward_ctrl
    modified_reward = original_reward - smooth_coef * ||a_t - a_{t-1}||^2
    """

    def __init__(self, env, smooth_coef: float = 0.05):
        super().__init__(env)
        self.smooth_coef = smooth_coef
        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        action = np.asarray(action, dtype=np.float32)

        if self.prev_action is None:
            smoothness_penalty = 0.0
            action_delta_norm = 0.0
        else:
            action_delta = action - self.prev_action
            action_delta_norm = float(np.linalg.norm(action_delta))
            smoothness_penalty = float(self.smooth_coef * np.sum(action_delta ** 2))

        modified_reward = float(reward - smoothness_penalty)

        self.prev_action = action.copy()

        info["original_reward"] = float(reward)
        info["smoothness_penalty"] = smoothness_penalty
        info["action_delta_norm"] = action_delta_norm
        info["modified_reward"] = modified_reward

        return obs, modified_reward, terminated, truncated, info


def make_modified_reacher_env(log_dir: Path | None = None, seed: int = 42, smooth_coef: float = 0.05):
    env = gym.make("Reacher-v5")
    env = ActionSmoothnessRewardWrapper(env, smooth_coef=smooth_coef)
    env.reset(seed=seed)

    if log_dir is not None:
        env = Monitor(
            env,
            filename=str(log_dir / "monitor.csv"),
            info_keywords=(
                "reward_dist",
                "reward_ctrl",
                "original_reward",
                "smoothness_penalty",
                "action_delta_norm",
                "modified_reward",
            ),
        )

    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth-coef", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    run_name = f"modified_smooth_{args.smooth_coef:g}"

    log_dir = project_root / "logs" / run_name
    model_dir = project_root / "models" / run_name
    result_dir = project_root / "results" / run_name

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=== Training PPO with Modified Reward on Reacher-v5 ===")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Seed: {args.seed}")
    print(f"Smoothness coefficient: {args.smooth_coef}")
    print(f"Device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    env = make_modified_reacher_env(
        log_dir=log_dir,
        seed=args.seed,
        smooth_coef=args.smooth_coef,
    )

    eval_env = make_modified_reacher_env(
        log_dir=None,
        seed=args.seed + 1,
        smooth_coef=args.smooth_coef,
    )
    eval_env = Monitor(eval_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=str(model_dir),
        name_prefix="ppo_reacher_modified_checkpoint",
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
        tb_log_name=run_name,
    )

    final_model_path = model_dir / "ppo_reacher_modified_final"
    model.save(str(final_model_path))

    mean_modified_reward, std_modified_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )

    # Fair task-performance evaluation on the original Reacher-v5 reward.
    original_eval_env = gym.make("Reacher-v5")
    original_eval_env.reset(seed=args.seed + 2)
    original_eval_env = Monitor(original_eval_env)

    mean_original_reward, std_original_reward = evaluate_policy(
        model,
        original_eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )

    print("\n=== Final Evaluation ===")
    print("Evaluation under modified reward:")
    print(f"Mean modified reward over 20 episodes: {mean_modified_reward:.3f}")
    print(f"Std modified reward over 20 episodes: {std_modified_reward:.3f}")

    print("\nEvaluation under original Reacher-v5 reward:")
    print(f"Mean original reward over 20 episodes: {mean_original_reward:.3f}")
    print(f"Std original reward over 20 episodes: {std_original_reward:.3f}")

    print(f"\nFinal model saved to: {final_model_path}.zip")

    env.close()
    eval_env.close()
    original_eval_env.close()


if __name__ == "__main__":
    main()