from unityagents import UnityEnvironment
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from lib.environments import UnityEnv
from lib.ddpg_agent import DDPG_Agent
from lib.td3_agent import TD3_Agent
from lib.replay_buffer import ReplayBuffer

import sys
import argparse

BUFFER_SIZE = int(1e6)  # MM replay buffer size
BATCH_SIZE = 128        # minibatch size - MM is 100
RANDOM_SEED = 2
LEARN_EVERY_STEPS = 1


def eval_episode(env, agents):
    states = env.reset()
    all_actions = np.zeros((len(agents), env.action_size))
    # print('all_actions.shape', all_actions.shape)
    steps_in_episode = 0
    while True:
        steps_in_episode += 1
        for i, agent in enumerate(agents):
            actions = agent.act(states[i], train=False)
            # print('actions.shape', actions.shape)
            all_actions[i,:] = actions
        states, rewards, dones, info = env.step(all_actions)
        # print('rewards ', rewards, 'dones', dones)
        ret = info[0]['episodic_return']
        if ret is not None:
            # print('dones', dones)
            break
    return ret, steps_in_episode

def eval_episodes(name, env, agents, num_episodes=20):
    total_rewards = np.zeros(num_episodes)
    record = pd.DataFrame(columns=['time', 'score'])
    max_steps_in_episode = 0
    for i in range(num_episodes):
        t0 = time.time()
        episode_rewards, steps = eval_episode(env, agents)
        if steps > max_steps_in_episode:
            max_steps_in_episode = steps
        # total_rewards.append(np.sum(episode_rewards))
        # print('rewards from episode', episode_rewards)
        total_rewards[i] = np.max(episode_rewards)
        print(f'Episode: {i+1} average so far {np.sum(total_rewards) / (i+1)}')
        # print(f'Episode: {i+1} rewards {episode_rewards} total_rewards {np.sum(total_rewards)} average so far {np.sum(total_rewards) / (i+1)}')
        t1 = time.time()
        record = record.append(dict(time=int(t1-t0),
                                    score=round(np.max(episode_rewards), 2)), ignore_index=True)
        record.to_csv(f'{name}-test-results.csv')
    print('Number of steps in longest', max_steps_in_episode)
    return np.mean(total_rewards)

def save_agents(agents, suffix=None):
    for i, agent in enumerate(agents):
        if suffix:
            agent.save(f'{agent.name}-{suffix}')
        else:
            agent.save()

def load_agents(agents, suffix=None):
    for i, agent in enumerate(agents):
        if suffix:
            agent.load(filename=f'{agent.name}-{suffix}')
        else:
            agent.load(filename=agent.name)

def train_agent(name, env, agents, max_episodes=5000, break_on_reward=10, save_interval=200, eval_interval=200):
    print(time.strftime("%H:%M:%S", time.localtime()), 'start training')
    states = env.reset()
    # print('state', states)
    record_avg = pd.DataFrame(columns=['time', 'episodes', 'average_score'])
    record_train = pd.DataFrame(columns=['time', 'episodes', 'score'])
    t0 = time.time()
    highest_episode_reward = 0
    highest_reward = 0
    episode_agent_rewards = np.zeros(len(agents))
    dones = [False, False]
    num_episodes = 0
    eval_num_episodes = 100
    while True:
        # print('save_interval', config.save_interval, 'total_steps', agent.total_steps)
        if any(dones):
            num_episodes += 1
            states = env.reset()
            max_reward_for_episode = np.max(episode_agent_rewards)
            t1 = time.time()
            record_train = record_train.append(dict(time=int(t1-t0),
                                                    episodes=int(num_episodes),
                                                    score=round(max_reward_for_episode, 2)), ignore_index=True)
            record_train.to_csv(f'{name}_training_scores.csv')
            if max_reward_for_episode > highest_episode_reward:
                highest_episode_reward = max_reward_for_episode
                print('new highest episode reward', highest_episode_reward)
            episode_agent_rewards = np.zeros(len(agents))
            if num_episodes > max_episodes:
                save_agents(agents)
                break
            if save_interval and not num_episodes % save_interval:
                save_agents(agents)
            if eval_interval and not num_episodes % eval_interval:
                print('------ Evaluate training -------')
                average_reward = eval_episodes(name, env, agents, num_episodes=eval_num_episodes)
                print(time.strftime("%H:%M:%S", time.localtime()),
                    f'After {agents[0].total_steps} steps ', 'average reward:', average_reward)
                t1 = time.time()
                record_avg = record_avg.append(dict(time=int(t1-t0),
                                                    episodes=int(num_episodes),
                                                    average_score=round(average_reward, 2)), ignore_index=True)
                record_avg.to_csv(f'{name}_average_for_{eval_num_episodes}_episodes.csv')
                if average_reward > highest_reward:
                    highest_reward = average_reward
                    save_agents(agents, suffix='best-so-far')
                    if highest_reward > break_on_reward:
                        break

        all_actions = np.zeros((len(agents), env.action_size))
        for i, agent in enumerate(agents):
            actions = agent.act(states[i])
            # print('actions.shape', actions.shape)
            all_actions[i,:] = actions
        # print('all_actions.shape', all_actions.shape)
        next_states, rewards, dones, info = env.step(all_actions)
        episode_agent_rewards += rewards
        for i, agent in enumerate(agents):
            # print('rewards', rewards)
            # print('states.shape', states.shape)
            # print('states[i].shape', states[i].shape)
            agent.step(states[i], all_actions[i,:], rewards[i], next_states[i], dones[i])
        states = next_states
    average_reward = eval_episodes(name, env, agents, num_episodes=eval_num_episodes)
    if average_reward > highest_reward:
        highest_reward = average_reward
        save_agents(agents, suffix='best-so-far')
    print(time.strftime("%H:%M:%S", time.localtime()),
            f'After {num_episodes} episodes ', 'average reward:', average_reward)
    t1 = time.time()
    record_avg = record_avg.append(dict(time=round(t1-t0),
                                        episodes=int(num_episodes),
                                        average_score=round(average_reward, 2)), ignore_index=True)
    record_avg.to_csv(f'{name}_average_for_{eval_num_episodes}_episodes.csv')

agent_fns = {
    'ddpg': DDPG_Agent,
    'td3': TD3_Agent,
}

if __name__ == '__main__':
    print(len(sys.argv))
    parser = argparse.ArgumentParser(
        description='This program trains and tests RL models'
    )
    parser.add_argument('-n', '--name', metavar='name', help='Used as basis to store any output')
    parser.add_argument('-m', '--mode', metavar='mode', help='train or test', default='train')
    parser.add_argument('-f', '--filename', metavar='filename', help='filename of model')
    parser.add_argument('-a', '--agent', metavar='agent', required=True, help='agent - ddpg, td3, a2c, ppo')
    parser.add_argument('-s', '--episodes', metavar='episodes', help='Number of episodes')
    parser.add_argument('-r', '--shared-replay', metavar='shared_replay', help='Use shared replay buffer', default=False, type=bool)
    parser.add_argument('-l', '--learn-frequency', metavar='learn_freq', help='Number of episodes between learning', default=LEARN_EVERY_STEPS, type=int)


    args = parser.parse_args()

    print(args.mode)
    print(args.filename)
    print(args.agent)
    train_mode = True if args.mode == 'train' else False
    if args.agent not in agent_fns:
        print('invalid agent, must be ddpg or td3')
        sys.exit()
    unity_game = os.environ.get('UNITY_GAME')
    env = UnityEnv('Tennis', unity_game, train_mode=train_mode)
    # env = UnityEnv('Tennis', './Tennis_Linux_NoVis/Tennis.x86_64', train_mode=train_mode)
    # env = UnityEnv('Tennis', './Tennis_Linux/Tennis.x86_64', train_mode=train_mode)
    name = args.name if args.name else args.agent
    agent_fn = agent_fns[args.agent]
    agents = []

    if args.shared_replay:
        replay_buffer = ReplayBuffer(env.action_size, BUFFER_SIZE, BATCH_SIZE, RANDOM_SEED)
    else:
        replay_buffer = False
    for i in range(env.num_agents):
        agent = agent_fn(name=f'{name}-{i}',
                         state_size=env.state_size,
                         action_size=env.action_size,
                         random_seed=RANDOM_SEED,
                         warm_up=int(1e3),
                         replay_buffer=replay_buffer,
                         buffer_size=BUFFER_SIZE,
                         batch_size=BATCH_SIZE,
                         learn_every_steps=args.learn_frequency)
        agents.append(agent)
    if train_mode:
        max_episodes = int(args.episodes) if args.episodes else int(3000)
        train_agent(args.name, env, agents, max_episodes=max_episodes)
    else:
        print('evaluating')
        load_agents(agents)
        n_episodes = args.steps if args.steps else 100
        average_reward = eval_episodes(args.name, env, agents, num_episodes=n_episodes)
        print(f'average score for {n_episodes} episodes is {average_reward}')


