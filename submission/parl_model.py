import paddle
import paddle.nn as nn
import parl
import copy
import paddle.nn.functional as F


class ParlModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(ParlModel, self).__init__()
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)

    def policy(self, obs):
    	return self.actor(obs)
    	
    def value(self, obs):
    	return self.critic(obs)


class Actor(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.fc_mean = nn.Linear(64, act_dim)
        self.log_std = paddle.static.create_parameter(
            [act_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0))

    def forward(self, obs):
        x = paddle.tanh(self.fc1(obs))
        x = paddle.tanh(self.fc2(x))

        mean = self.fc_mean(x)
        return mean, self.log_std


class Critic(parl.Model):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs):
        x = paddle.tanh(self.fc1(obs))
        x = paddle.tanh(self.fc2(x))
        value = self.fc3(x)

        return value
