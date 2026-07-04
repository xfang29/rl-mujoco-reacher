import gymnasium as gym
import torch


def main():
    print("=== Python / PyTorch Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    print("\n=== Gymnasium / MuJoCo / Reacher-v5 Check ===")
    env = gym.make("Reacher-v5")

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nSample action: {action}")
    print(f"Next observation shape: {obs.shape}")
    print(f"Reward after one random step: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")

    env.close()


if __name__ == "__main__":
    main()