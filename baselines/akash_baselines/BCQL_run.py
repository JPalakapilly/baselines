from BCQL import BCQ
from custom_envs import BehavSimEnv
from BCQL_Memory import ReplayBuffer
import matplotlib.pyplot as plt
import numpy as np

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def train_BCQ(response_type_str):
    if(response_type_str == 'threshxxold_exp'):
        env = BehavSimEnv(response='t', one_day=True)
        env2 = BehavSimEnv(response='t', one_day=False)
    elif(response_type_str == 'sin'):
        env = BehavSimEnv(response='s',one_day=True)
        env2 = BehavSimEnv(response='w', one_day=False)
    elif(response_type_str == 'mixed'):
        env = BehavSimEnv(response='m',one_day=True)
        env2 = BehavSimEnv(response='m', one_day=False)
    else:
        env = BehavSimEnv(response='l',one_day=True)
        env2 = BehavSimEnv(response='l', one_day=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer()
    agent = BCQ(state_dim, action_dim,0,10,replay_buffer)
    collect_data_steps = 30

    state = env.prices[0]
    for step in range(collect_data_steps):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        useless = env2.step(action)
        replay_buffer.add((state,action,next_state,reward,done))
        state = next_state

    rewards = []
    rewards2 = []
    action_star = []
    max_iters = 30
    for training_iterations in range(max_iters):
        print("=========================")
        print("Training Iteration " + str(training_iterations) + " out of " + str(max_iters))
        print("")
        agent.train(100)
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        replay_buffer.add((state,action,next_state,reward,done))
        rewards.append(reward)
        useless_next, reward2, useless_done, useless_info = env2.step(action)
        rewards2.append(reward2)
        action_star = action
        print("=========================")

    plt.figure()
    plt.plot(rewards, label='reward')
    plt.plot(moving_average(rewards),label='Moving Avg')
    plt.title("Rewards of new SAC V2 (Trained One-Day " + response_type_str, pad = 20.0)
    plt.legend()
    plt.xlabel("Day Number")
    plt.xticks([i for i in range(len(rewards)) if i % 10 == 0], labels = [i for i in range(len(rewards)) if i % 10 == 0])
    plt.ylabel("Reward")
    plt.savefig(response_type_str + '_training.png')

    plt.figure()
    plt.plot(rewards2, label='true reward')
    plt.title("Daily Response (Trained on One_Day Hourly" + response_type_str, pad = 20.0)
    plt.legend()
    plt.xlabel("Day Number")
    plt.xticks([i for i in range(len(rewards)) if i % 10 == 0], labels = [i for i in range(len(rewards)) if i % 10 == 0])
    plt.ylabel("Reward")
    plt.savefig(response_type_str + '_results.png')

    # plt.figure()
    # plt.plot(min_combined_losses)
    # plt.xlabel('Iteration ')
    # plt.ylabel("Combined Q1+Q2 loss")
    # plt.title("Combined Critic Loss at each Day", pad = 20.0)
    # plt.savefig(response_type_str + '_min_q_loss.png')
    return 

train_BCQ('threshold_exp')
train_BCQ('sin')
train_BCQ('linear')
train_BCQ('mixed')


