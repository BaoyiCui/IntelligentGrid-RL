# -*- coding: UTF-8 -*-
import copy

import numpy as np
# from Agent.DoNothingAgent import DoNothingAgent
# from Agent.RandomAgent import RandomAgent
from submission.agent import Agent
from Environment.base_env import Environment
from utilize.settings import settings
import paddle
from parl.utils import ReplayMemory


WARMUP_STEPS = 100
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 20
ACT_DIM = 54
OBS_DIM = 620


def run_train_episode(agent, env, rpm):
    action_dim = ACT_DIM
    obs = env.reset()
    done = False
    episode_reward, episode_steps = 0, 0
    reward = 0.0
    episode_max_steps = 288
    
    while not done:
        episode_steps += 1
        if rpm.size() < WARMUP_STEPS:
            action_temp = np.random.uniform(-1, 1, size=action_dim)
        else:
            action_temp = agent.act(obs, reward, done)
        action = agent._process_action(obs, action_temp)    
          
        next_obs, reward, done, _ = env.step(action)
        terminal = float(done) if episode_steps < episode_max_steps else 0
        obs_temp = agent._process_obs(obs)
        next_obs_temp = agent._process_obs(next_obs)
        action_array = []
        
        rpm.append(obs_temp, action_temp, reward, next_obs_temp, terminal)
        obs = next_obs
        episode_reward += reward
        
        if rpm.size() >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
            agent.agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal)
            
    return episode_reward, episode_steps
    


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
    max_timestep = 10  # 最大时间步数
    max_episode = 2  # 回合数
    submission_path = './submission'
    model_path = './submission/saved_model/model-1'
    # my_agent = RandomAgent(settings.num_gen)
    agent = Agent(copy.deepcopy(settings), submission_path)
    env = Environment(settings, 'EPRIReward')
    rpm = rpm = ReplayMemory(max_size=MEMORY_SIZE, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    max_score = 0.0
    print('-------train-------')
    for i in range(1000):
        episode_reward, episode_steps = run_train_episode(agent, env, rpm)
        print('episode:', i,'episode_steps:', episode_steps, '    episode_reward:',  episode_reward)
        # print('episode_steps:', episode_steps)
        if i % 100 ==0 :
            episode_max_steps = 288
            scores = []
            score = run_one_episode(env, SEED, 0, episode_max_steps, agent)
            scores.append(score)
            print('score:', score)
            if score >= max(scores) :
                paddle.save(agent.model.state_dict(), model_path)
    # print('-------test-------')
    # # episode_max_steps = 288
    # # scores = []
    # # for start_idx in np.random.randint(settings.num_sample, size=20):
    # #     score = run_one_episode(env, SEED, start_idx, episode_max_steps, agent)
    # #     scores.append(score)
    # #     print('score:', score)
    # paddle.save(agent.model.state_dict(), model_path)
    # print('scores:', score)
    # run_task(my_agent)
