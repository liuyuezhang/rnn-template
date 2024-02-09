import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Elman(nn.Module):
    def __init__(self, in_size, hid_size, nonlinear='relu-tanh', alpha=1., bias=True):
        super(Elman, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.nonlinear = nonlinear
        self.alpha = alpha
        if self.nonlinear == 'relu':
            self.sigma = F.relu
        elif self.nonlinear == 'tanh':
            self.sigma = F.tanh
        elif self.nonlinear == 'relu-tanh':
            self.sigma = lambda x: F.relu(F.tanh(x))

        self.u = None  # (B, N)
        
        W = torch.empty((self.hid_size, self.hid_size), dtype=torch.float32)
        nn.init.xavier_uniform(W)
        self.W = Parameter(W)
        
        b = torch.empty(self.hid_size, dtype=torch.float32)
        nn.init.zeros_(b)
        self.b = Parameter(b, requires_grad=bias)
        
        W_in = torch.empty((self.in_size, self.hid_size), dtype=torch.float32)
        nn.init.xavier_uniform(W_in)
        self.W_in = Parameter(W_in)

    def init(self, u):
        self.u = u

    def detach(self):
        self.u = torch.Tensor(self.u.data)

    def input(self, x):
        return x @ self.W_in

    def dynamics(self, h):
        return self.sigma(self.u @ self.W + self.b + h)

    def forward(self, x):
        h = self.input(x)
        r = self.dynamics(h)
        self.u = (1. - self.alpha) * self.u + self.alpha * r
        return r


class RNNReadout(nn.Module):
    def __init__(self, N, init_size, in_size, out_size, model='elman', alpha=1., bias=True, bias_out=False, norm=False,
                 nonlinear='relu', init_weight=False, noise_type='none', noise_scale=0.0, lr=1e-4, reg='l2', coef_neuron=0., coef_rnn=0.):
        super(RNNReadout, self).__init__()
        self.N = N
        self.init_size = init_size
        self.in_size = in_size
        self.out_size = out_size
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.loss_fn = nn.MSELoss()
        self.coef_neuron = coef_neuron
        self.coef_rnn = coef_rnn
        self.init_weight = init_weight

        self.rnn = Elman(in_size=in_size, hid_size=N, nonlinear=nonlinear, alpha=alpha, bias=bias, norm=norm)
        self.W_init = nn.Linear(init_size, N, bias=bias_out)
        self.W_out = nn.Linear(N, out_size, bias=bias_out)

        # optimizers
        self.optimizer = torch.optim.Adam(lr=lr)
        self.optimizer.setup(self)

        # loss
        self.loss = 0.
        self.neuron = 0.
        self.loss_out = 0.
        self.T = 0

    def clear_loss(self):
        self.loss = 0.
        self.T = 0

    def get_init(self, s0):
        # inverse transformation
        if self.init_weight:
            u = self.W_init(s0)
        else:
            u = s0 @ self.W_out.W
        return u

    def init(self, s0):
        # init rnn
        u = self.get_init(s0)
        self.rnn.init(u)
        # clear loss
        self.clear_loss()
        return self.rnn.u

    def forward(self, a, noise=None):
        # rnn
        r = self.rnn.forward(a, noise=noise, noise_scale=self.noise_scale)
        y = self.W_out(r)
        return r, y

    def step(self, a, s, train=True):
        with torch.set_grad_enabled(train):
            # forward
            r, y = self.forward(a)

            # loss
            self.loss += self.loss_fn(y, s)
            self.T += 1
        return r, y

    def update(self):
        # main loss
        loss = self.loss / self.T

        # backward and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # detach
        self.rnn.detach()
        self.clear_loss()
        return loss.item()
