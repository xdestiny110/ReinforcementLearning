from env_maze import MazeEnv

env = MazeEnv()
for _ in range(5):
    d = False
    while d == False:
        s = -1
        while s == -1:
            act = env.action_space.sample()
            s,r,d,p = env.step(act)
        env.render()
    env.reset()
