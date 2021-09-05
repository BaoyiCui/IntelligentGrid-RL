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

    def sample(self, obs):
        """ Sample action from current policy given observation

        Args:
            obs (np.array): observation
        """
        obs = paddle.to_tensor(obs)
        value, action, action_log_probs = self.alg.sample(obs)

        return value.detach().numpy(), action.detach().numpy(), \
            action_log_probs.detach().numpy()


    def predict(self, obs):
        obs = paddle.to_tensor(obs, dtype='float32')
        action = self.alg.predict(obs)

        return action.detach().numpy()

    def learn(self, next_value, gamma, gae_lambda, ppo_epoch, num_mini_batch,
              rollouts):
        """ Learn current batch of rollout for ppo_epoch epochs.

        Args:
            next_value (np.array): next predicted value for calculating advantage
            gamma (float): the discounting factor
            gae_lambda (float): lambda for calculating n step return
            ppo_epoch (int): number of epochs K
            num_mini_batch (int): number of mini-batches
            rollouts (RolloutStorage): the rollout storage that contains the current rollout
        """
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(ppo_epoch):
            data_generator = rollouts.sample_batch(next_value, gamma,
                                                   gae_lambda, num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, \
                    value_preds_batch, return_batch, old_action_log_probs_batch, \
                            adv_targ = sample

                obs_batch = paddle.to_tensor(obs_batch)
                actions_batch = paddle.to_tensor(actions_batch)
                value_preds_batch = paddle.to_tensor(value_preds_batch)
                return_batch = paddle.to_tensor(return_batch)
                old_action_log_probs_batch = paddle.to_tensor(
                    old_action_log_probs_batch)
                adv_targ = paddle.to_tensor(adv_targ)

                value_loss, action_loss, dist_entropy = self.alg.learn(
                    obs_batch, actions_batch, value_preds_batch, return_batch,
                    old_action_log_probs_batch, adv_targ)

                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                dist_entropy_epoch += dist_entropy

        num_updates = ppo_epoch * num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
    def value(self, obs):
        """ Predict value from current value function given observation

        Args:
            obs (np.array): observation
        """
        obs = paddle.to_tensor(obs)
        val = self.alg.value(obs)

        return val.detach().numpy()

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
        
