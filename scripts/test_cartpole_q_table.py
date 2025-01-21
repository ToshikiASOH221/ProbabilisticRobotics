import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

q_table = np.loadtxt("../data/opt_q_table.csv", delimiter=",")
print("q table:\n{}".format(q_table))

state = 0
actions = [0, 1]
alpha = 0.1
gamma = 0.99
reward = 0
steps = 5000
episodes = 3 # 試行回数
learn_convergence = 195 # 学習収束の基準値（報酬の値）
learn_proc_rec_flag = 100 # 学習経過の様子を定期的に記録
record_flag = False
learned_flag = False
ep_finish_flag = False

# 状態パラメータの範囲
digitize_num = 4

pos_range = [-2.4, 2.4] #[-4.8, 4.8]
vel_range = [-3.0, 3.0] #[-10, 10]
ang_range = [-0.418, 0.418]
omg_range = [-2.0, 2.0] #[-10, 10]

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

class Q():
    global q_table
    
    def __init__(self):
        #self.env = gym.make("CartPole-v1", render_mode="human")
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.env = gym.wrappers.RecordVideo(self.env, video_folder="../figs/opt", episode_trigger=lambda x: True, disable_logger=True)
        self.env.reset()
        self.q_table = q_table
        
    def state(self):
        digitized = [
            np.digitize(self.obs[0], bins=bins(pos_range[0], pos_range[1], digitize_num)),
            np.digitize(self.obs[1], bins=bins(vel_range[0], vel_range[1], digitize_num)),
            np.digitize(self.obs[2], bins=bins(ang_range[0], ang_range[1], digitize_num)),
            np.digitize(self.obs[3], bins=bins(omg_range[0], omg_range[1], digitize_num))
        ]
        state = sum([x*(digitize_num**i) for i, x in enumerate(digitized)])
        #print("[debug] state:{}".format(state))
        
        return state
        
    def action(self, next_state):
        next_action = np.argmax(self.q_table[next_state,:])
        
        return next_action
    
        
    def run(self):
        global ep_finish_flag, learned_flag, record_episodes
 
        # エピソードの繰り返し
        for ep in range(episodes):
            # 環境の初期化
            self.obs = np.array(self.env.reset()[0])
            state = self.state()
            action = np.argmax(self.q_table[state])
            
            # 1エピソード内のステップの繰り返し
            for step in range(steps):
                
                self.obs, _, terminated, truncated, info = self.env.step(action)
                #print("obs:{}".format(self.obs))
                
                # terminated or trucated = True の時：ステップ上限に達したか，倒れたか
                if terminated or truncated:
                    break
                
                # Q値表の更新
                next_state = self.state() # 状態を離散化
                
                # 次の行動の決定
                action = self.action(next_state)
                
                state = next_state
                
                #print("[episode-step:{}-{}]".format(ep, step))
                
                if ep_finish_flag:
                    ep_finish_flag = False
                    previous_ep_steps = step
                    break
   
def main():
    q_learning = Q()
    q_learning.run()
    

if __name__ == "__main__":
    main()