# -*- coding: UTF-8 -*-
import copy
from collections import deque
import numpy as np
from submission.agent import Agent
from submission.storage import RolloutStorage
from Environment.base_env import Environment
from utilize.settings import settings
import paddle
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

NUM_STEPS = 288
ENTROPY_COEF = 0.
ACT_DIM = 54
OBS_DIM = 620
LR = 3e-4
GAMMA = 0.99
EPS = 1e-5
GAE_LAMBDA = 0.95
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NROM = 0.5
PPO_EPOCH = 10
CLIP_PARAM = 0.2
BATCH_SIZE = 32

def run_train_episode(agent, env, rollouts):
    action_dim = ACT_DIM
    obs = env.reset()
    done = False
    

    return episode_reward, episode_steps, critic_loss, actor_loss


def run_one_episode(env, seed, start_idx, episode_max_steps, agent):
    # print('start_index:', start_idx)
    # obs = env.reset(seed=seed, start_sample_idx=start_idx)
    obs = env.reset()
    reward = 0.0
    done = False
    sum_reward = 0.0
    sum_steps = 0.0
    for step in range(episode_max_steps):
        action = agent.act(obs, reward, done)
        action = agent._process_action(obs, action)
        obs, reward, done, info = env.step(action)
        sum_reward += reward
        sum_steps += 1
        if done:
            break
    return sum_reward, sum_steps


if __name__ == "__main__":
    SEED = 0
    submission_path = './submission'
    model_path = './submission/saved_model/model-1'
    env = Environment(settings, 'EPRIReward')
    agent = Agent(copy.deepcopy(settings), submission_path)
    rollouts = RolloutStorage(NUM_STEPS, 620, 54)                              
    max_score = 0.0
    obs = env.reset()

    rollouts.obs[0] = np.copy(agent._process_obs(obs))
    episode_rewards = deque(maxlen=10)
    num_updates = int(10e5) // NUM_STEPS
    episode_num = 0
    num = []
    rewards = []
    critic_loss_s = []
    actor_loss_s = []
    print('-------train-------')

    for i in range(num_updates):
        episode_reward = 0.0
        for step in range(NUM_STEPS):
            # Sample actions
            value, action, action_log_prob = agent.act(rollouts.obs[step])
            #Obser reward and next obs
            obs, reward, done, info = env.step(agent._process_action(obs, action))
            
            if done:
                print('episode{} done'.format(i))
            
            # print('info:', info)
            # If done then clean the history of obs
            masks = paddle.to_tensor(
                    [[0.0]] if done else [[1.0]],
                    dtype='float32'
                    )
            # bad_masks = paddle.to_tensor([[0.0]], dtype='float32')
            bad_masks = paddle.to_tensor(
                    [[0.0]] if 'fail_info' in info.keys() else [[1.0]], 
                    dtype='float32'
                    )
            rollouts.append(agent._process_obs(obs), action, action_log_prob, value, reward, masks, bad_masks)
            next_value = agent.agent.value(rollouts.obs[-1])
            value_loss, action_loss, dist_entropy = agent.agent.learn(next_value, GAMMA, GAE_LAMBDA, PPO_EPOCH, BATCH_SIZE, rollouts)
            # print(reward)
            rollouts.after_update()
            episode_reward += reward

            # print('episode:', episode_num)
            print('step_reward:', reward)
            print('episode_reward:', episode_reward)
            if done:
                episode_num = episode_num + 1
                
                env.reset()
                break
