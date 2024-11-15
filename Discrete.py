import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gym.make(
    "FrozenLake-v1",
    desc=generate_random_map(size=8),
    map_name="8x8",
    is_slippery=True,
    render_mode="human",
)

for episode in range(10):
    observation, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0.0
    step_count = 0

    while not terminated or truncated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

    print(
        f"Episode {episode + 1} finished after {step_count} steps with reward {episode_reward}"
    )
    print(f"Terminated: {terminated}, Truncated: {truncated}")

env.close()
