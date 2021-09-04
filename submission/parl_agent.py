import os
import paddle
import numpy as np
import copy
from submission.parl_model import *
from submission.parl_algorithm import *
import parl
from utilize.settings import settings


from abc import abstractmethod

def wrap_action(adjust_gen_p):
    act = {
        'adjust_gen_p': adjust_gen_p,
        'adjust_gen_v': np.zeros_like(adjust_gen_p)
    }
    return act

class ParlAgent(parl.Agent):
    
    def __init__(self, algoritTAUhm, act_dim, expl_noise=0.1):

        super(ParlAgent, self).__init__(algorithm)
        self.act_dim = act_dim
        self.expl_noise = expl_noise
        self.alg.sync_target(decay=(1.0-TAU))

    def sample(self, obs_features, obs):
        # obs_features = paddle.to_tensor(obs_features, dtype='float32')
        prob = self.predict(obs_features)
        # prob = prob.numpy()
        action_noise = np.random.normal(0, self.expl_noise, size=self.act_dim)
        act = (prob + action_noise).clip(-1, 1)
        #act = self._process_action(obs, act)
        return act

    def predict(self, obs):

        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action_numpy = self.alg.predict(obs)
        # prob = self.alg.predict(obs)
        # act = self._process_action()
        re_act = action_numpy.argmax().numpy()[0]

        return re_act

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        action = paddle.to_tensor(action, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        # print('c_loss: ', critic_loss.numpy(), 'act_loss', actor_loss.numpy())
        return critic_loss, actor_loss
        # act = np.expand_dims(act, axis=-1)
        # reward = np.expand_dims(reward, axis=-1)
        # obs = paddle.to_tensor(obs, dtype='float32')
        # act = paddle.to_tensor(act, dtype='int32')
        # reward = paddle.to_tensor(reward, dtype='float32')
        #
        # loss = self.alg.learn(obs, act, reward)

        # return loss.numpy()[0]

    def _process_action(self, obs, action):
        N = len(action)
        Settings = copy.deepcopy(settings)
        gen_p_action_space = obs.action_space['adjust_gen_p']

        low_bound = gen_p_action_space.low
        high_bound = gen_p_action_space.high

        mapped_action = low_bound + (action - (-1.0)) * (
                (high_bound - low_bound) / 2.0)
        mapped_action[0, Settings.balanced_id] = 0.0
        mapped_action = np.clip(mapped_action, low_bound, high_bound)

        return wrap_action(mapped_action)
        
