import gymnasium as gym
import numpy as np

class InvertedPendulumPIDController:
    def __init__(self, kp=50.0, ki=10.0, kd=10.0):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        
        # Integral error storage
        self.integral_error = 0.0
        self.prev_error = 0.0
        
        # Anti-windup limits for integral term
        self.integral_limit = 5.0
        
        # Time step for integration
        self.dt = 0.05  # MuJoCo default timestep
    
    def compute_action(self, state):
        # State space for InvertedPendulum-v4:
        # state[0] -> Angular position (theta)
        # state[1] -> Angular velocity (theta_dot)
        # state[2] -> x position
        # state[3] -> x velocity
        
        theta = state[0]  # Angular position
        theta_dot = state[1]  # Angular velocity
        
        # Calculate errors
        error = theta  # Error relative to upright position (0 radians)
        
        # Update integral term with anti-windup
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        
        # Calculate derivative term
        derivative = theta_dot
        
        # Compute PID control action
        action = (-self.kp * error 
                 -self.ki * self.integral_error 
                 -self.kd * derivative)
        
        # MuJoCo InvertedPendulum action space is [-3, 3]
        return np.clip(action, -3, 3)
    
    def reset(self):
        """Reset integral term between episodes"""
        self.integral_error = 0.0
        self.prev_error = 0.0

def run_inverted_pendulum(custom_init=None):
    env = gym.make('InvertedPendulum-v4', render_mode='human')
    controller = InvertedPendulumPIDController()
    
    episode_rewards = []
    
    for episode in range(5):
        observation, info = env.reset()
            
        # Reset controller's integral term
        controller.reset()
        total_reward = 0
        
        for t in range(1000):
            # Get action from controller
            action = controller.compute_action(observation)
            
            # Apply action and get new state
            observation, reward, terminated, truncated, info = env.step([action])
            total_reward += reward
            
            # Print detailed state info every 100 steps
            if t % 100 == 0:
                print(f"Step {t}: "
                      f"Angle = {np.rad2deg(observation[0]):.2f}°, "
                      f"Action = {action:.2f}, "
                      f"Integral = {controller.integral_error:.3f}")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    
    env.close()
    print(f"\nAverage Reward over {len(episode_rewards)} episodes: {np.mean(episode_rewards):.2f}")


if __name__ == "__main__":
    run_inverted_pendulum()
    