import gymnasium as gym


def check_env(env_id: str):
    print(f"\n=== Checking {env_id} ===")

    env = gym.make(env_id)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation sample: {obs[:5]} ...")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"One-step reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")

    env.close()


def main():
    envs = [
        "Hopper-v5",
        "Walker2d-v5",
        "HalfCheetah-v5",
    ]

    for env_id in envs:
        try:
            check_env(env_id)
        except Exception as e:
            print(f"\nFailed to check {env_id}")
            print(e)


if __name__ == "__main__":
    main()