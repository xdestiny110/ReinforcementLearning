#-*- coding:utf-8 -*-'

# 参考https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
# 和https://github.com/openai/gym/blob/master/gym/envs/toy_text/discrete.py
import gym
import numpy as np

class MazeEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Discrete(4*4) #状态空间，代表处于4*4棋盘的某个格子中
        self.action_space = gym.spaces.Discrete(4) #动作集合，即上下左右
        self.rewards = dict(zip([x for x in range(4*4)], [0 for _ in range(4*4)])) #到达某个状态的回报
        self.rewards[6] = self.rewards[9] = -1
        self.rewards[10] = 1
        self.terminate_states = [6,9,10]
        

