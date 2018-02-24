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
        
        actions_list = list(self.env.P[self.env.s].keys()) # 依据状态转移表获得该状态下有效的动作
        #动作选取策略是采用贪婪还是随机
        if np.random.uniform() < self.e_greedy:            
            state_action = self.q_table[self.env.s]
            state_action_sorted = sorted(state_action.items(), key=lambda a : a[1]) # 依据Q-value进行排序
            for k, _ in state_action_sorted:
                if actions_list[k] != []:
                    action = k
                    break
        else:
            action = self.env.action_space.sample()
            while actions_list[action] == []:
                action = self.env.action_space.sample()
        return action

    def learn(self, last_state, state, action, reward, is_terminted):
        q_predict = self.q_table[last_state][action]
        q_target = reward
        if is_terminted == False:
            q_target += self.reward_decay*np.max(self.q_table[state])[1]
        self.q_table[last_state][action] += self.lr*(q_target-q_predict)