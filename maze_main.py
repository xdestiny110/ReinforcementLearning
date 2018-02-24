from Maze.env_maze import MazeEnv
from RLAlgorithm.QLearning import QLearning
import time

env = MazeEnv()
qlearning = QLearning(env)
for round in range(100):
    print("\nRound = %d"%(round))
    print("==================")
    is_terminted = False
    env.reset()
    while not is_terminted:
        act = qlearning.choose_action()
        state, reward, is_terminted, _ = env.step(act)
        qlearning.learn(env.laststate, env.s, act, reward, is_terminted)
        # env.render()
        # time.sleep(0.04)

print("================")
print("solve")
env.reset()
env.render()
is_terminted = False
while not is_terminted:
    act = qlearning.choose_best_action()
    print(env.dir[act])
    _, _, is_terminted, _ = env.step(act)
    env.render()
    time.sleep(1)

print("================")
print("q_table:")
for i in range(16):
    if i in qlearning.q_table:
        print("state #{} = {}".format(i, qlearning.q_table[i]))
    else:
        print("state #{} = {0:0,1:0,2:0,3:0}".format(i))

