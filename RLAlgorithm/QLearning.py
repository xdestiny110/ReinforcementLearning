#-*- coding:utf-8 -*-'
import numpy as np

class QLearning():
    def __init__(self, env, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        self.lr = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        self.q_table = {}
        self.env = env

    def choose_action(self):
        if self.env.s not in self.q_table:
            self.q_table[self.env.s] = {x : 0 for x in range(self.env.action_space.n)}
        
        actions_list = self.env.P[self.env.s] # 依据状态转移表获得该状态下有效的动作
        
        #动作选取策略是采用贪婪还是随机
        if np.random.uniform() < self.e_greedy:
            # 去掉无效的动作
            state_action = [(k,v) for k,v in self.q_table[self.env.s].items() if actions_list[k] != []] 
            # Q-value有可能相同，因此先打乱再使用max
            np.random.shuffle(state_action)
            action = max(state_action, key = lambda x: x[1])[0]
        else:
            action = self.env.action_space.sample()
            while actions_list[action] == []:
                action = self.env.action_space.sample()
        return action

    def learn(self, last_state, state, action, reward, is_terminted):
        q_predict = self.q_table[last_state][action]
        q_target = reward
        if is_terminted == False:
            q_target += self.reward_decay*np.max(self.q_table[last_state])[1]
        self.q_table[last_state][action] += self.lr*(q_target-q_predict)