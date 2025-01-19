import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo 


state = 0
actions = [0, 1]
alpha = 0.5
gamma = 0.9
reward = 0
steps = 200
episodes = 2000 # 試行回数
record_flag = False
learned_flag = False

# 状態パラメータの範囲
digitize_num = 6
pos_bins = np.linspace(-2.4, 2.4, digitize_num)
vel_bins = np.linspace(-3, 3, digitize_num)
ang_bins = np.linspace(-41.8, 41.8, digitize_num)
omg_bins = np.linspace(-2, 2, digitize_num)
print("bins:{}".format(np.array([pos_bins,vel_bins,ang_bins,omg_bins])))

# 状態パラメータを離散化
def states_digitize(observetion):
    digitized = [
        np.digitize(observetion[0], bins=pos_bins),
        np.digitize(observetion[1], bins=vel_bins),
        np.digitize(observetion[2], bins=ang_bins),
        np.digitize(observetion[3], bins=omg_bins),
    ]
    state = sum([x*(digitize_num**i) for i, x in enumerate(digitized)])
    
    return state

# Qテーブルのヒートマップを描画
def visualized_qtable_heatmap(q_table):
    return

class Q():
    def __init__(self, ):
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.env = gym.wrappers.RecordVideo(self.env, video_folder="./", episode_trigger=record_flag, disable_logger=True)
        self.env.reset()
        self.q_table = np.random.uniform(-1, 1, (digitize_num**4, len(actions)))
        
    def action(self):
        return
    
    def reward(self):
        return
    
    def updateQ(self):
        updated_q = max()
        
        
    def run(self):
        # エピソードの繰り返し
        for ep in range(episodes):
            obs = self.env.reset()
            
            # 1エピソード内のステップの繰り返し
            for step in steps:
                if learned_flag:
                    self.env.render()
        
        
        self.env.close()
    

def main():
    q_learning = Q()
    q_learning.run()
    

if __name__ == "__main__":
    main()