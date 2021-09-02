import os
import paddle
import numpy as np
import copy
from submission.parl_model import *
from submission.parl_algorithm import *
import parl

import copy
from abc import abstractmethod

class ParlAgent(parl.Agent):
    
    def __init__(self, algorithm):

        super(ParlAgent, self).__init__(algorithm)

    def sample(self, obs):

        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        prob = prob.numpy()
        print(type(prob))
        prob = prob.reshape(-1)
        act = np.random.choice(len(prob), 1, p=prob)[0]

        return act

    def predict(self, obs):

        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        act = prob.argmax().numpy()[0]

        return act

    def learn(self, obs, act, reward):

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')

        loss = self.alg.learn(obs, act, reward)

        return loss.numpy()[0]

        
