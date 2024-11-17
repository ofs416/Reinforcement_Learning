import numpy as np
import gymnasium as gym

from custom_frozen_lake import CustomFrozenLake


class TemporalQLearning:
    def __init__(self, env, lambda_discount=1.0):
        self.env = env
        self.lambda_discount = lambda_discount
        self.aplha_lr = 1.0
        self.epsilon_init = 0.95
        self.epsilon_final = 0.1

        # Initialize value function to zeros - no prior knowledge
        self.q_table = np.zeros(env.unwrapped.desc.shape, env.unwrapped.actions.shape)

    def policy(self, state, epsilon):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            return np.random.randint(self.env.action_space.n)
        return np.argmax(self.q_table[state])

    def update(self, step_data):
        state, action, next_state, reward = *step_data
        self.q_table[state, action] = (1 - self.aplha_lr) * self.q_table[state, action] 
        + self.alpha_lr * (reward + self.lambda_discount * np.max(self.q_table[next_state]))


    def train(self, episodes=1000):
        for episode in range(episodes):
            state, _ = self.env.reset()

            done = False
            episode_reward = 0

            while not done:
                action = self.policy(state, self.epsilon_init)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.update((state, action, next_state, reward))
                done = terminated or truncated
                episode_reward += reward
                state = next_state


if __name__ == "__main__":
    # Create base environment and wrap it
    base_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env = CustomFrozenLake(
        base_env, hole_reward=-10.0, step_reward=-1, goal_reward=100.0
    )

    # Create and train agent
    agent = TemporalQLearning(env)
    agent.train(episodes=10_000)

    # Print the environment layout for reference
    print("\nEnvironment Layout:")
    print(env.unwrapped.desc.astype(str))

    env.close()
