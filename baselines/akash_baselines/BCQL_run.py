from BCQL import BCQ
from BCQL_Memory import ReplayBuffer

import sys
sys.path.append("..")

from behavioral_sim.custom_envs import BehavSimEnv
from behavioral_sim.custom_envs import HourlySimEnv
import matplotlib.pyplot as plt
import numpy as np

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def train_BCQ(response_type_str='mixed', extra_train = 1):
    if(response_type_str == 'threshxold_exp'):
        env = HourlySimEnv(response='t', one_day=False, energy_in_state=True, yesterday_in_state=False,
                            day_of_week = True)
        # env2 = BehavSimEnv(response='t', one_day=False)
    elif(response_type_str == 'sin'):
        env = HourlySimEnv(response='s', one_day=False, energy_in_state=True, yesterday_in_state=False,
                            day_of_week = True)
        # env2 = BehavSimEnv(response='w', one_day=False)
    elif(response_type_str == 'mixed'):
        env =HourlySimEnv(response='m', one_day=False, energy_in_state=True, yesterday_in_state=False,
                            day_of_week = True)
        # env2 = BehavSimEnv(response='m', one_day=False)
    else:
        env = HourlySimEnv(response='t', one_day=False, energy_in_state=True, yesterday_in_state=False,
                            day_of_week = True)
        # env2 = BehavSimEnv(response='l', one_day=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer()
    agent = BCQ(state_dim, action_dim,0,10,replay_buffer)
    collect_data_steps = 30

    state = None
    state_flag = True
    while (env.day <= 30):
        print("=========================")
        print("DAY: " + str(env.day) + " | Hour : " + str(env.hour))
        print("")
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        # useless = env2.step(action)
        if(not state_flag):
            replay_buffer.add((state,action,next_state,reward,done))
        else:
            state_flag = False
        state = next_state

    rewards = []
    rewards2 = []
    action_star = []
    # max_iters = 30
    while(env.day <= 60):
        print("=========================")
        print("DAY: " + str(env.day) + " | Hour : " + str(env.hour))
        print("")
        agent.train(extra_train)
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        replay_buffer.add((state,action,next_state,reward,done))
        rewards.append(reward)
        # useless_next, reward2, useless_done, useless_info = env2.step(action)
        # rewards2.append(reward2)
        action_star = action
        print("=========================")
    print("DONE")
    # plt.figure()
    # plt.plot(rewards, label='reward')
    # plt.plot(moving_average(rewards),label='Moving Avg')
    # plt.title("Rewards of new SAC V2 (Trained One-Day " + response_type_str, pad = 20.0)
    # plt.legend()
    # plt.xlabel("Day Number")
    # plt.xticks([i for i in range(len(rewards)) if i % 10 == 0], labels = [i for i in range(len(rewards)) if i % 10 == 0])
    # plt.ylabel("Reward")
    # plt.savefig(response_type_str + '_training.png')

    # plt.figure()
    # plt.plot(rewards2, label='true reward')
    # plt.title("Daily Response (Trained on One_Day Hourly" + response_type_str, pad = 20.0)
    # plt.legend()
    # plt.xlabel("Day Number")
    # plt.xticks([i for i in range(len(rewards)) if i % 10 == 0], labels = [i for i in range(len(rewards)) if i % 10 == 0])
    # plt.ylabel("Reward")
    # plt.savefig(response_type_str + '_results.png')

    # plt.figure()
    # plt.plot(min_combined_losses)
    # plt.xlabel('Iteration ')
    # plt.ylabel("Combined Q1+Q2 loss")
    # plt.title("Combined Critic Loss at each Day", pad = 20.0)
    # plt.savefig(response_type_str + '_min_q_loss.png')
    return rewards

def train_curve_finder(max_iter, response_type_str=None):
    def train_store_rewards(response_type_str=None):
        sampled_days = [19,16,29,18,14,23,9,21,10,30]
        #Key = Day | Val = list for SAC Reward
        rewards_dict = {i-1: [] for i in sampled_days}
        # rewards_list_no_e = []
        # rewards_list_no_e_min = []
        # rewards_list_no_e_max = []
        for iteration in range(51,max_iter,10):
            #Add error bounds, just for loop then return avg, pointwise-max/min
            # max_reward = -1e10
            # cum_reward = 0
            # min_reward = 0
            curr_rewards_bcq = train_BCQ(response_type_str = response_type_str, 
                                        extra_train=iteration)
            print(curr_rewards_bcq)
            for day in rewards_dict.keys():
                rewards_dict[29].append(curr_rewards_bcq)
                continue
            # if(curr_total_rewards > max_reward):
            #     max_reward = curr_total_rewards
            # if(curr_total_rewards < min_reward):
            #     min_reward = curr_total_rewards
            # cum_reward += curr_total_rewards

            # rewards_list_e.append(cum_reward / 5)
            # rewards_list_e_min.append(min_reward)
            # rewards_list_e_max.append(max_reward)
        # return np.array(rewards_list_e), np.array(rewards_list_e_min), np.array(rewards_list_e_min)
        
        return rewards_dict
    bcq_rewards_dict = train_store_rewards(response_type_str='mixed')

    sac_rewards_iter_day = np.array(bcq_rewards_dict[29])
    np.save("BCQ_rewards2", sac_rewards_iter_day)

# train_BCQ('threshold_exp')
# train_BCQ('sin')
# train_BCQ('linear')
train_curve_finder(101,'mixed')


