#-*- coding:utf-8 -*-'

# 参考https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
# 和https://github.com/openai/gym/blob/master/gym/envs/toy_text/discrete.py
from gym import Env, spaces

class MazeEnv(Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(4*4) #状态空间，代表处于4*4棋盘的某个格子中
        self.action_space = spaces.Discrete(4) #动作集合

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
        self.dir = {0:'Left', 1:'Right', 2:'Down', 3:'Up'}
        
        self.lastaction=None
        self.laststate=None
        self.s = 0
        self.viewer = None
        self.times = 0

        self.agent_view_trans = None
        self.screen_width = self.screen_height = 650
        self.unit_width = self.unit_height = 150
        self.offset_width = (self.screen_width-self.unit_width*4)/2
        self.offset_height = (self.screen_height-self.unit_height*4)/2

        self.seed()
        self.reset()

    def _check_terminate(self, state):
        if (state == 6) or (state == 9) or (state == 10):
            return True
        return False

    def _get_reward(self, state):
        if (state == 6) or (state == 9):
            return -100
        elif state == 10:
            return 100
        return -1

    def _get_loc(self):
        return self.s // 4, self.s % 4

    def seed(self, seed=None):
        pass

    def reset(self):
        # 一定从右下角作为起始点
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
        if self.viewer is None:
            # 注意绘制坐标系原点在左下角
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            lines = []
            for i in range(4):
                lines.append(rendering.Line((self.offset_width+i*self.unit_width, self.offset_height),(self.offset_width+i*self.unit_width, self.screen_height-self.offset_height)))
                lines[-1].set_color(0,0,0)
                lines.append(rendering.Line((self.offset_width, self.offset_height+i*self.unit_height),(self.screen_width-self.offset_width, self.offset_height+i*self.unit_height)))
                lines[-1].set_color(0,0,0)

            traps = []
            traps.append(rendering.make_circle(50))
            traps[-1].add_attr(rendering.Transform(translation=(1.5*self.unit_width+self.offset_width,2.5*self.unit_height+self.offset_height)))
            traps[-1].set_color(1,0,0)
            traps.append(rendering.make_circle(50))
            traps[-1].add_attr(rendering.Transform(translation=(2.5*self.unit_width+self.offset_width,1.5*self.unit_height+self.offset_height)))
            traps[-1].set_color(1,0,0)

            gold = rendering.make_circle(50)
            gold.add_attr(rendering.Transform(translation=(2.5*self.unit_width+self.offset_width,2.5*self.unit_height+self.offset_height)))
            gold.set_color(1,1,0)

            for v in lines:
                self.viewer.add_geom(v)
            for v in traps:
                self.viewer.add_geom(v)
            self.viewer.add_geom(gold)

            agent_view = rendering.make_circle(50)
            ix, iy = self._get_loc()
            self.agent_view_trans = rendering.Transform()
            agent_view.add_attr(self.agent_view_trans)
            agent_view.set_color(0,1,0)
            self.viewer.add_geom(agent_view)

        print("step = %d, current state = %d"%(self.times, self.s))
        ix, iy = self._get_loc()
        self.agent_view_trans.set_translation(newx = (0.5+ix)*self.unit_width+self.offset_width, newy = (0.5+iy)*self.unit_height+self.offset_height)
        return self.viewer.render(return_rgb_array = mode == 'rgb_array')
