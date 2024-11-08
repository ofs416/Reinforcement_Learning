import numpy as np
import gymnasium as gym

from custom_frozen_lake import CustomFrozenLake

import numpy as np
import gymnasium as gym


class MonteCarloAgent:
    def __init__(self, env, lambda_discount=1.0):
        self.env = env
        self.lambda_discount = lambda_discount

        # Initialize value function to zeros - no prior knowledge
        self.value_func = np.zeros(env.unwrapped.desc.shape)
        
        # Initialize returns for each state
        self.returns = {(i, j): [] for i in range(self.value_func.shape[0]) 
                                 for j in range(self.value_func.shape[1])}
        self.N = {(i, j): 0 for i in range(self.value_func.shape[0]) 
                           for j in range(self.value_func.shape[1])}
    
    def update(self, episode):
        states, actions, rewards = zip(*episode)
        
        # Calculate R_k as per notes
        R = 0  
        returns = []
        for r in reversed(rewards):  
            R = r + self.lambda_discount * R
            returns.insert(0, R)
            
        # Update value function for all states
        for t in range(len(states)):
            state = states[t]
            R = returns[t]
            self.N[state] += 1
            
            # Equation (2) from the notes
            self.value_func[state[0]][state[1]] = (
                (self.N[state] - 1) / self.N[state] * self.value_func[state[0]][state[1]] + 
                1/self.N[state] * R
            )
            
            self.returns[state].append(R)
    
    def train(self, episodes=1000):
        for episode in range(episodes):
            state, _ = self.env.reset()
            state_row = state // self.env.unwrapped.ncol
            state_col = state % self.env.unwrapped.ncol
            
            done = False
            episode_data = []
            episode_reward = 0
            
            while not done:
                action = np.random.randint(4)  
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Get coordinates of next state
                next_row = next_state // self.env.unwrapped.ncol
                next_col = next_state % self.env.unwrapped.ncol
                
                # Add current state, action, reward to episode data
                episode_data.append(((state_row, state_col), action, reward))
                
                # If terminal state reached, add it with its reward
                if done:
                    episode_data.append(((next_row, next_col), None, reward))
                
                state = next_state
                state_row = next_row
                state_col = next_col
            
            self.update(episode_data)


if __name__ == "__main__":
    # Create base environment and wrap it
    base_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    env = CustomFrozenLake(base_env, hole_reward=-10.0, step_reward=-1, goal_reward=100.0)
    
    # Create and train agent
    agent = MonteCarloAgent(env)
    agent.train(episodes=10_000)
    
    print("\nFinal Value Function:")
    print(agent.value_func)
    
    # Print the environment layout for reference
    print("\nEnvironment Layout:")
    print(env.unwrapped.desc.astype(str))
    
    env.close()