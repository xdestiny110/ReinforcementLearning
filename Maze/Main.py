import sys
sys.path.append("../")
from Maze.env_maze import MazeEnv
from RLAlgorithm.QLearning import QLearning

env = MazeEnv()
qlearning = QLearning(env)
for _ in range(5):
    is_terminted = False
    env.reset()
    while not is_terminted:
        act = qlearning.choose_action()
        state, reward, is_terminted,_ = env.step(act)
        qlearning.learn(env.laststate, env.s, act, reward, is_terminted)
