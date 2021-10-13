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
        ret = info[0]['episodic_return']
        if ret is not None:
            # print('dones', dones)
            break
    return ret, steps_in_episode

def eval_episodes(env, agents, num_episodes=20):
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
        print(f'Episodes: {i} average {np.sum(total_rewards) / i}')
        t1 = time.time()
        record = record.append(dict(time=round(t1-t0),
                                    score=round(np.sum(episode_rewards), 2)), ignore_index=True)
        record.to_csv(f'{agents[0].name}-test-results.csv')
    print('max_steps_in_episode', max_steps_in_episode)
    return np.mean(total_rewards)

def save_agents(agents, suffix=None):
    for i, agent in enumerate(agents):
        if suffix:
            agent.save(f'{agent.name}-{suffix}')
        else:
            agent.save()

def train_agent(name, env, agents, max_steps=1e6, break_on_reward=35, save_interval=1e4, eval_interval=1e4):
    print(time.strftime("%H:%M:%S", time.localtime()), 'start training')
    states = env.reset()
    # print('state', states)
    record = pd.DataFrame(columns=['time', 'steps', 'average_score'])
    t0 = time.time()
    highest_reward = 0
    dones = [False, False]
    while True:
        # print('save_interval', config.save_interval, 'total_steps', agent.total_steps)
        if any(dones):
            states = env.reset()
        if save_interval and not agents[0].total_steps % save_interval:
            save_agents(agents)
        if eval_interval and not agents[0].total_steps == 0 and not agents[0].total_steps % eval_interval:
            print('agent.eval_episodes')
            average_reward = eval_episodes(env, agents, num_episodes=100)
            print(time.strftime("%H:%M:%S", time.localtime()),
                  f'After {agents[0].total_steps} steps ', 'average reward:', average_reward)
            t1 = time.time()
            record = record.append(dict(time=round(t1-t0),
                                        steps=agent.total_steps,
                                        average_score=round(average_reward, 2)), ignore_index=True)
            record.to_csv(f'{name}.csv')
            if average_reward > highest_reward:
                highest_reward = average_reward
                save_agents(agents, suffix='best-so-far')
                if highest_reward > break_on_reward:
                    break

        if max_steps and agents[0].total_steps >= max_steps:
            # print('agent.close')
            save_agents(agents)
            # env.close()
            break
        all_actions = np.zeros((len(agents), env.action_size))
        for i, agent in enumerate(agents):
            actions = agent.act(states[i])
            # print('actions.shape', actions.shape)
            all_actions[i,:] = actions
        # print('all_actions.shape', all_actions.shape)
        next_states, rewards, dones, info = env.step(all_actions)
        for i, agent in enumerate(agents):
            # print('rewards', rewards)
            # print('states.shape', states.shape)
            # print('states[i].shape', states[i].shape)
            agent.step(states[i], all_actions[i,:], rewards[i], next_states[i], dones)
        states = next_states
    average_reward = eval_episodes(env, agents)
    if average_reward > highest_reward:
        highest_reward = average_reward
        save_agents(agents, suffix='best-so-far')
    print(time.strftime("%H:%M:%S", time.localtime()),
            f'After {agents[0].total_steps} steps ', 'average reward:', average_reward)
    t1 = time.time()
    record = record.append(dict(time=round(t1-t0),
                                steps=agents[0].total_steps,
                                average_score=round(average_reward, 2)), ignore_index=True)
    for agent in agents:
        record.to_csv(f'{agent.name}.csv')

agents = {
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
    parser.add_argument('-s', '--steps', metavar='steps', help='Number of steps')


    args = parser.parse_args()

    print(args.mode)
    print(args.filename)
    print(args.agent)
    train_mode = True if args.mode == 'train' else False
    if args.agent not in agents:
        print('invalid agent, must be ddpg or td3')
        sys.exit()
    unity_game = os.environ.get('UNITY_GAME')
    env = UnityEnv('Tennis', unity_game, train_mode=train_mode)
    # env = UnityEnv('Tennis', './Tennis_Linux_NoVis/Tennis.x86_64', train_mode=train_mode)
    # env = UnityEnv('Tennis', './Tennis_Linux/Tennis.x86_64', train_mode=train_mode)
    name = args.name if args.name else args.agent
    agent_fn = agents[args.agent]
    agents = []

    replay_buffer = ReplayBuffer(env.action_size, BUFFER_SIZE, BATCH_SIZE, RANDOM_SEED)
    for i in range(env.num_agents):
        agent = agent_fn(name=f'name-{i}',
                         state_size=env.state_size,
                         action_size=env.action_size,
                         random_seed=RANDOM_SEED,
                         warm_up=int(1e4),
                         replay_buffer=replay_buffer,
                         buffer_size=BUFFER_SIZE,
                         batch_size=BATCH_SIZE,
                         learn_every_steps=LEARN_EVERY_STEPS)
        agents.append(agent)
    if train_mode:
        max_steps = int(args.steps) if args.steps else int(1e6)
        train_agent(args.name, env, agents, max_steps=max_steps)
    else:
        print('evaluating')
        agent.load(filename=args.name)
        n_episodes = args.steps if args.steps else 100
        average_reward = eval_episodes(env, agents, num_episodes=n_episodes)
        print(f'average score for {n_episodes} episodes is {average_reward}')


