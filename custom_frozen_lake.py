import gymnasium as gym


# Create a custom environment by inheriting from FrozenLake-v1
class CustomFrozenLake(gym.Wrapper):
    def __init__(self, env, hole_reward=-1.0, step_reward=-0.1, goal_reward=1.0):
        super().__init__(env)
        self.hole_reward = hole_reward
        self.step_reward = step_reward
        self.goal_reward = goal_reward

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Modify rewards based on the outcome
        if terminated:
            # Check if agent reached the goal (G) or fell in a hole (H)
            state_row = observation // self.unwrapped.ncol
            state_col = observation % self.unwrapped.ncol
            if self.unwrapped.desc[state_row][state_col] == b"G":
                reward = self.goal_reward
            else:  # Must have fallen in a hole
                reward = self.hole_reward
        else:
            # If not terminated, it was a regular step
            reward = self.step_reward

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    # Create the base environment
    base_env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="8x8",
        is_slippery=True,
        render_mode="human",
    )

    # Wrap it with custom rewards
    env = CustomFrozenLake(
        base_env,
        hole_reward=-10.0,  # Penalty for falling in a hole
        step_reward=-0.01,  # Small penalty for each step to encourage faster completion
        goal_reward=10.0,  # Reward for reaching the goal
    )

    # Max steps based on environment size - using unwrapped env
    max_steps = 100 if env.unwrapped.desc.shape[0] == 4 else 200

    # Run episodes with custom rewards
    for episode in range(1):
        observation, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        step_count = 0

        while not (terminated or truncated) and step_count < max_steps:
            env.render()
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

        print(f"\nEpisode {episode + 1} finished after {step_count} steps")
        print(f"Total reward: {episode_reward:.1f}")
        print(f"Terminated: {terminated}, Truncated: {truncated}\n")

    env.close()
