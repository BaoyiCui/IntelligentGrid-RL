import paddle
import paddle.nn as nn
import parl
import copy
import paddle.nn.functional as F


class ParlModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(ParlModel, self).__init__()
        self.actor_model = Actor(obs_dim, act_dim)
        self.critic_model = Critic(obs_dim, act_dim)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class Actor(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, action_dim)

    def forward(self, obs):
        a = F.relu(self.l1(obs))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        return paddle.tanh(self.l4(a))
        # return self.l3(a)


class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(obs_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256+action_dim,128)
        # self.l2 = nn.Linear(512, 256)
        self.l4 = nn.Linear(128, 1)

    def forward(self, obs, action):
        q = F.relu(self.l1(obs))
        q = F.relu(self.l2(q))
        q = F.relu(self.l3(paddle.concat([q, action], 1)))
        #q = F.relu(self.l2(q))
        return self.l4(q)