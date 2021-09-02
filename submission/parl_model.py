import paddle
import paddle.nn as nn
import parl
import copy
import paddle.nn.functional as F

class ParlModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(ParlModel, self).__init__()
        self.l1 = nn.Linear(obs_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.mean_linear = nn.Linear(256, act_dim)

    def forward(self, x):
        x1 = F.relu(self.l1(x))
        print(x1.shape)
        x2 = F.relu(self.l2(x1))
        print(x2.shape)
        act_mean = self.mean_linear(x2)
        action = paddle.tanh(act_mean)
        return action
