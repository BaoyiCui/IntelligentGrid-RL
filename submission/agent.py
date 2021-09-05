import os
import paddle
import numpy as np
# from submission.grid_model import GridModel
# from submission.grid_agent import GridAgent
import  parl.algorithms
from parl.algorithms import PPO
from submission.parl_agent import *
from submission.parl_model import *
from submission.parl_algorithm import *


import copy
from abc import abstractmethod


OBS_DIM = 620
ACT_DIM = 54

LR = 3e-4
GAMMA = 0.99
EPS = 1e-5  # Adam optimizer epsilon (default: 1e-5)
GAE_LAMBDA = 0.95  # Lambda parameter for calculating N-step advantage
ENTROPY_COEF = 0.  # Entropy coefficient (ie. c_2 in the paper)
VALUE_LOSS_COEF = 0.5  # Value loss coefficient (ie. c_1 in the paper)
MAX_GRAD_NROM = 0.5  # Max gradient norm for gradient clipping
NUM_STEPS = 2048  # data collecting time steps (ie. T in the paper)
PPO_EPOCH = 10  # number of epochs for updating using each T data (ie K in the paper)
CLIP_PARAM = 0.2  # epsilon in clipping loss (ie. clip(r_t, 1 - epsilon, 1 + epsilon))
BATCH_SIZE = 32

# Logging Params
LOG_INTERVAL = 1


class BaseAgent():
    def __init__(self, num_gen):
        self.num_gen = num_gen

    def reset(self, ons):
        pass

    @abstractmethod
    def act(self, obs, reward, done=False):
        pass

def wrap_action(adjust_gen_p):
    act = {
        'adjust_gen_p': adjust_gen_p,
        'adjust_gen_v': np.zeros_like(adjust_gen_p)
    }
    return act




class Agent(BaseAgent):

    def __init__(self, settings, this_directory_path):
        self.settings = settings
        
        model_path = os.path.join(this_directory_path, "saved_model/model-1")
        # model = GridModel(OBS_DIM, ACT_DIM)
        self.model = ParlModel(OBS_DIM, ACT_DIM)
        self.alg = PPO(self.model, CLIP_PARAM, VALUE_LOSS_COEF, ENTROPY_COEF, LR, EPS,MAX_GRAD_NROM)
        self.agent = ParlAgent(self.alg)
        
    def act(self, rollouts_obs):
        value, action, action_log_prob = self.agent.sample(rollouts_obs)
        return value, action, action_log_prob
    
    def _process_obs(self, obs):
        # loads
        loads = []
        loads.append(obs.load_p)
        loads.append(obs.load_q)
        loads.append(obs.load_v)
        loads = np.concatenate(loads)

        # prods
        prods = []
        prods.append(obs.gen_p)
        prods.append(obs.gen_q)
        prods.append(obs.gen_v)
        prods = np.concatenate(prods)
        
        # rho
        rho = np.array(obs.rho) - 1.0
        
        features = np.concatenate([loads, prods, rho.tolist()])
        return features
    
    def _process_action(self, obs, action):
        N = len(action)
        gen_p_action_space = obs.action_space['adjust_gen_p']

        low_bound = gen_p_action_space.low
        high_bound = gen_p_action_space.high

        mapped_action = low_bound + (action - (-1.0)) * (
            (high_bound - low_bound) / 2.0)
        mapped_action = mapped_action.reshape(-1)
        mapped_action[self.settings.balanced_id] = 0.0
        mapped_action = np.clip(mapped_action, low_bound, high_bound)
        
        return wrap_action(mapped_action)
