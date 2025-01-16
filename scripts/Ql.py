import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo 

steps = 300
state = [0, 0, 0, 0]
action = [0, 1]
alpha = 0.1
epsilon = 0.9

env = gym.make("CartPole-v1", render_mode="rgb_array")
print("actions:{}".format(env.action_space))

trigger = lambda t: t % 10 == 0 #動画を保存をするエピソードの指定
env = gym.wrappers.RecordVideo(env, video_folder="./", episode_trigger=trigger, disable_logger=True)
env.reset()
    
def run(steps):
    for i in range(steps):
        env.render()
        #action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action[0])
        print("[{}] {}".format(i, obs))
        
        #if terminated  or truncated:
        #    break
        
    env.close()
    

def main():
    run(steps)
    

if __name__ == "__main__":
    main()