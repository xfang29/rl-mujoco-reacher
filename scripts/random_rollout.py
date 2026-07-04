import gymnasium as gym


def main():
    env = gym.make("Reacher-v5", render_mode="human")

    obs, info = env.reset(seed=42)

    total_reward = 0.0
    num_steps = 500

    for step in range(num_steps):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    print(f"Random rollout finished.")
    print(f"Total reward over {num_steps} steps: {total_reward:.3f}")
    print(f"Average reward per step: {total_reward / num_steps:.3f}")


if __name__ == "__main__":
    main()