import numpy as np
import gymnasium as gym

from custom_frozen_lake import CustomFrozenLake


class TemporalQLearning:
    def __init__(self, env):
        self.env = env
        self.lambda_discount = 1.0
        self.alpha_lr = 1.0
        self.epsilon_init = 0.5
        self.epsilon_final = 0.0

        # Initialize value function to zeros - no prior knowledge
        self.q_table = np.zeros((self.env.unwrapped.observation_space.n, 
        self.env.unwrapped.action_space.n))


    def policy(self, state, step_total):
        # Epsilon-greedy action selection
        epsilon = self.epsilon_final + (self.epsilon_init - self.epsilon_final) * np.exp(-0.1 * step_total)
        if np.random.random() < epsilon:
            return self.env.unwrapped.action_space.sample()
        elif np.max(self.q_table[state]) == np.min(self.q_table[state]):
               return self.env.unwrapped.action_space.sample()
        return np.argmax(self.q_table[state])
                

    def update(self, step_data):
        state, action, next_state, reward = step_data
        # Q(s,a) = (1-α)Q(s,a) + α[R + γ*max(Q(s',a'))]
        self.q_table[state, action] = (1 - self.alpha_lr) * self.q_table[state, action] + \
            self.alpha_lr * (reward + self.lambda_discount * np.max(self.q_table[next_state]))

    def train(self, episodes):
        step_total = 0

        for episode in range(episodes):
            state, _ = self.env.reset()
            
            done = False
            episode_reward = 0

            while not done:
                action = self.policy(state, step_total)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.update((state, action, next_state, reward))
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                step_total += 1

            print(f"Episode: {episode}, Reward: {episode_reward}")
            #self.print_q_table_max()
            #self.print_q_table_min()
        
        print(self.q_table)

    def print_q_table_max(self):
        # Assuming the environment is a 4x4 grid
        grid_size = 4
        state_table = np.zeros((grid_size, grid_size), dtype=int)

        for state in range(grid_size * grid_size):
            row = state // grid_size
            col = state % grid_size
            state_table[row, col] = np.max(self.q_table[state])

        print("State Table with Max of Q-table:")
        print(state_table)
    
    def print_q_table_min(self):
        # Assuming the environment is a 4x4 grid
        grid_size = 4
        state_table = np.zeros((grid_size, grid_size), dtype=int)

        for state in range(grid_size * grid_size):
            row = state // grid_size
            col = state % grid_size
            state_table[row, col] = np.min(self.q_table[state])

        print("State Table with Min of Q-table:")
        print(state_table)
            



if __name__ == "__main__":
    # Create base environment and wrap it
    env = CustomFrozenLake(
        hole_reward=-1.0, 
        step_reward=-0.01,  
        goal_reward=1.0,  
        map_name="4x4",
        render_mode="ansi",
        is_slippery=False
    )

    # Create and train agent
    agent = TemporalQLearning(env)
    agent.train(episodes=32)

    # Print the environment layout for reference
    print("\nEnvironment Layout:")
    print(env.unwrapped.desc.astype(str))

    env.close()
