import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm


class EnvMnist():
    def __init__(self, batch_size=50, train=True, shuffle=True):
        self.batch_size = 50
        self.ob_space = (self.batch_size, 1, 28, 28)
        # data
        dataset = datasets.MNIST('./data',  train=train, transform=transforms.ToTensor(), download=True)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    def reset(self):
        return next(iter(self.dataloader))
    
    def step(self):
        return torch.zeros(self.ob_space, dtype=np.float32)
    

# # start of trial
# init_state = env.reset()
# model.init(init_state)

# for t in trange(1, T+1):
#     state = env.step()
#     y = model.step(state)

#     loss = (y, init_state)


env = EnvMnist()
for i in tqdm(range(10000)):
    env.reset()