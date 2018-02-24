#-*- coding:utf-8 -*-'

# 参考https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
# 和https://github.com/openai/gym/blob/master/gym/envs/toy_text/discrete.py
from gym import Env, spaces

class MazeEnv(Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(4*4) #状态空间，代表处于4*4棋盘的某个格子中
        self.action_space = spaces.Discrete(4) #动作集合，即上下左右

        self.P = {s : {a : [] for a in range(4)} for s in range(4*4)} #状态转移表
        for s in range(4*4):
            if s > 3:
                self.P[s][0].append((1, s-4, self._get_reward(s-4), self._check_terminate(s-4))) #分别代表转移概率，下一个状态，回报，是否结束
            if s < 12:
                self.P[s][1].append((1, s+4, self._get_reward(s+4), self._check_terminate(s+4)))
            if s%4 != 0:
                self.P[s][2].append((1, s-1, self._get_reward(s-1), self._check_terminate(s-1)))
            if s%4 != 3:
                self.P[s][3].append((1, s+1, self._get_reward(s+1), self._check_terminate(s+1)))
        self.dir = {0:'Up', 1:'Down', 2:'Left', 3:'Right'}
        
        self.lastaction=None
        self.laststate=None
        self.s = 0
        self.viewer = None
        self.times = 0

        self.seed()
        self.reset()

    def _check_terminate(self, state):
        if (state == 6) or (state == 9) or (state == 10):
            return True
        return False

    def _get_reward(self, state):
        if (state == 6) or (state == 9):
            return -1
        elif state == 10:
            return 1
        return 0

    def seed(self, seed=None):
        pass

    def reset(self):
        # 一定从左上角作为起始点
        self.s = 0
        self.lastaction = None
        self.laststate = None
        self.times = 0
        return self.s

    def step(self, action):
        if self.P[self.s][action] == []:
            return (-1,0,False,{})
        p, s, r, d = self.P[self.s][action][0]
        self.laststate = self.s
        self.s = s
        self.lastaction = action
        self.times = self.times+1
        return (s, r, d, {"prob" : p})

    def render(self, mode='human'):
        # screen_width = screen_height = 600
        # scale = screen_height/4

        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(screen_width, screen_height)
        print("step = %d, current state = %d"%(self.times, self.s))
