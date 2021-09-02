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
    
    def __init__(self, algorithm):

        super(ParlAgent, self).__init__(algorithm)

    def sample(self, obs_features, obs):
        obs_features = paddle.to_tensor(obs_features, dtype='float32')
        #print(obs)
        prob = self.alg.predict(obs_features)
        prob = prob.numpy()
        #print(type(prob))
        #prob = prob.reshape(-1)
        #print(prob)
        #act = np.random.choice(len(prob), 1, p=prob)[0]
        act = prob
        #act = self._process_action(obs, act)
        return act

    def predict(self, obs):

        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        act = self._process_action()
        re_act = prob.argmax().numpy()[0]

        return re_act

    def learn(self, obs, act, reward):

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')

        loss = self.alg.learn(obs, act, reward)

        return loss.numpy()[0]

    def _process_action(self, obs, action):
        N = len(action)
        Settings = copy.deepcopy(settings)
        gen_p_action_space = obs.action_space['adjust_gen_p']

        low_bound = gen_p_action_space.low
        high_bound = gen_p_action_space.high

        mapped_action = low_bound + (action - (-1.0)) * (
                (high_bound - low_bound) / 2.0)
        print(mapped_action)
        print(Settings.balanced_id)
        mapped_action[0, Settings.balanced_id] = 0.0
        mapped_action = np.clip(mapped_action, low_bound, high_bound)

        return wrap_action(mapped_action)
        
