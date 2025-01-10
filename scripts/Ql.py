import numpy as np
import gymnasium as gym
#np.bool = np.bool_

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

def main():
    for i in range(1000):
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

if __name__ == "__main__":
    main()