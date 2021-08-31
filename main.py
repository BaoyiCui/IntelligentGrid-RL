# -*- coding: UTF-8 -*-
import copy

import numpy as np
# from Agent.DoNothingAgent import DoNothingAgent
# from Agent.RandomAgent import RandomAgent
from submission.agent import Agent
from Environment.base_env import Environment
from utilize.settings import settings


def run_one_episode(env, seed, start_idx, episode_max_steps, agent):
    print('start_index:', start_idx)
    obs = env.reset(seed=seed, start_sample_idx=start_idx)
    reward = 0.0
    done = False
    sum_reward = 0.0
    sum_steps = 0.0
    for step in range(episode_max_steps):
        action = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(action)
        sum_reward += reward
        sum_steps += 1
        if done:
            break
    return sum_reward, sum_steps


# def run_task(my_agent):
#     for episode in range(max_episode):
#         print('------ episode ', episode)
#         env = Environment(settings, "EPRIReward")
#         print('------ reset ')
#         obs = env.reset()
#         reward = 0.0
#         done = False
#         # while not done:
#         for timestep in range(max_timestep):
#             ids = [i for i, x in enumerate(obs.rho) if x > 1.0]
#             # print("overflow rho: ", [obs.rho[i] for i in ids])
#             print('------ step ', timestep)
#             action = my_agent.act(obs, reward, done)
#             # print("adjust_gen_p: ", action['adjust_gen_p'])
#             # print("adjust_gen_v: ", action['adjust_gen_v'])
#             obs, reward, done, info = env.step(action)
#             print('info:', info)
#             if done:
#                 break


if __name__ == "__main__":
    SEED = 0
    max_timestep = 10  # 最大时间步数
    max_episode = 1  # 回合数
    submission_path = './submission'
    # my_agent = RandomAgent(settings.num_gen)
    agent = Agent(copy.deepcopy(settings), submission_path)
    env = Environment(settings)
    episode_max_steps = 288
    scores = []

    for start_idx in np.random.randint(settings.num_sample, size=20):
        score = run_one_episode(env, SEED, start_idx, episode_max_steps, agent)
    print('score:', score)
    # run_task(my_agent)
