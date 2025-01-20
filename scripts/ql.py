import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo 


state = 0
actions = [0, 1]
alpha = 0.1
gamma = 0.99
reward = 0
steps = 200
episodes = 100000 # 試行回数
learn_convergence = 190 # 学習収束の基準値（報酬の値）
learn_proc_rec_flag = 100 # 学習経過の様子を定期的に記録
record_flag = False
learned_flag = False
ep_finish_flag = False

# 動画レンダリング処理
record_flag = lambda t: t in [0, 10 , 100, 200, 500, 1000, 2000]
# 状態パラメータの範囲
digitize_num = 6

pos_range = [-4.8, 4.8]
vel_range = [-10, 10]
ang_range = [-0.418, 0.418]
omg_range = [-10, 10]

# Qテーブルのヒートマップを描画
def qtable_heatmap(q_tables):
    min = np.min(q_tables)
    max = np.max(q_tables)
    norm_q_tables = (q_tables - min) / (max - min)
    
    return norm_q_tables

def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

class Q():
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.env = gym.wrappers.RecordVideo(self.env, video_folder="../figs", episode_trigger=record_flag, disable_logger=True)
        self.env.reset()
        self.q_table = np.random.uniform(0, 0, (digitize_num**4, len(actions)))
        
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
        
    def action(self, next_state, episode):
        epsilon = 0.5*(1/(episode+1))
        if epsilon <= np.random.uniform(0,1):
            next_action = np.argmax(self.q_table[next_state,:])
        else:
            next_action = np.random.choice([0,1])
        
        return next_action
    
    def reward(self, step, previous_ep_steps):
        if step < steps-5: # 倒れたか範囲外の位置に遷移したか
            reward = -100
            #reward -= (abs(self.obs[0])*10)**2
            #reward -= (abs(self.obs[2])*10)**2
            #print("[debug] pole position reward:{}".format(-(abs(self.obs[0])*10)**2))
            #print("[debug] pole angle reward:{}".format(-(abs(self.obs[2])*10)**2))
        else: # ステップ上限まで立ち続けた
            reward = 1
            
        # 振り子が寄り垂直に近いほど高い報酬
        #reward += abs((pos_range[1] - abs(self.obs[0]))*5)
        #reward += ((ang_range[1] - abs(self.obs[2]))*10)**2
        #print("[debug] pole position reward:{}".format(abs((pos_range[1] - abs(self.obs[0])*5))))
        #print("[debug] pole angle reward:{}".format(((0.418 - abs(self.obs[2]))*10)**2))
        
        return reward
    
    def updateQ(self, state, action, next_state, reward):
        max_q = max(self.q_table[next_state, :]) # 次状態の価値の最大値を取得
        self.q_table[state, action] = (1-alpha)*self.q_table[state, action] + alpha*(reward + gamma*max_q) # Q値表の更新
        
        
    def run(self):
        global ep_finish_flag, record_flag
        
        ep_rewards = np.array([])
        previous_ep_steps = 0
        # エピソードの繰り返し
        for ep in range(episodes):
            # 環境の初期化
            self.obs = np.array(self.env.reset()[0])
            state = self.state()
            action = np.argmax(self.q_table[state])
            ep_reward = 0 # 一度の試行の報酬を格納
            
            # 1エピソード内のステップの繰り返し
            for step in range(steps):
                # 学習終了処理
                if learned_flag:
                    self.env.render(render_mode="human")
                
                self.obs, _, terminated, truncated, info = self.env.step(action)
                #print("obs:{}".format(self.obs))
                
                # terminated or trucated = True の時：ステップ上限に達したか，倒れたか
                if terminated or truncated:
                    reward = self.reward(step, previous_ep_steps)
                    ep_finish_flag = True
                else:
                    reward = 1 # 各ステップで立っていたら報酬
                
                ep_reward += reward
                
                # Q値表の更新
                next_state = self.state() # 状態を離散化
                self.updateQ(state, action, next_state, reward)
                
                # 次の行動の決定
                action = self.action(next_state, ep)
                
                state = next_state
                
                #print("[episode-step:{}-{}]".format(ep, step))
                
                if ep_finish_flag:
                    ep_finish_flag = False
                    previous_ep_steps = step
                    break
        
            #print("[debug] q-table:\n{}".format(self.q_table))
            print("episode steps:{}".format(step))
            print("[{}] episode reward:{}".format(ep, ep_reward))
            
            ep_rewards = np.append(ep_rewards, ep_reward)
            print("[{}] total reward mean:{}".format(ep, np.mean(ep_rewards)))
            
            # 学習終了判定・処理
            if ep < 100 and ep_rewards[ep-100:ep].mean() > learn_convergence:
                record_flag = True
                self.env.close()
                break
    

def main():#
    q_learning = Q()
    q_learning.run()
    

if __name__ == "__main__":
    main()