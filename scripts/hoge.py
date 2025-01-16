import gymnasium as gym
from gymnasium.wrappers import RecordVideo 

env = gym.make('CartPole-v1', render_mode="rgb_array")
trigger = lambda t: t % 10 == 0 #動画を保存をするエピソードの指定
env = gym.wrappers.RecordVideo(env, video_folder="./", episode_trigger=trigger, disable_logger=True)
env.reset()

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()