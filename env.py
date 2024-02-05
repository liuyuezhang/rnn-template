import numpy as np


class Env():
    def __init__(self):
        self.dataset = None
        # self.state = None
        pass

    def reset(self):
        return state
    
    def step(self):
        return next_state
    

# start of trial
init_state = env.reset()
model.init(init_state)

for t in trange(1, T+1):
    state = env.step()
    y = model.step(state)

    loss = (y, init_state)
