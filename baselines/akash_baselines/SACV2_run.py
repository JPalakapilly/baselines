from Memory import ReplayMemory
from SACV2 import SoftActorCritic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append("..")
import behavioral_sim
from behavioral_sim.custom_envs import BehavSimEnv
from behavioral_sim.custom_envs import HourlySimEnv

replay_size = 10000


total_numsteps = 60
start_steps = 30
batch_size = 10
action_star = None

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def train(response_type_str, extra_train, energy=False, day_of_week=False):
    if(response_type_str == 'threshold_exp'):
        #env = HourlySimEnv(response='t', one_day=True, energy_in_state=False)
        env2 = HourlySimEnv(response='t', one_day=False, energy_in_state=energy, yesterday_in_state=False,
                            day_of_week = day_of_week)
    elif(response_type_str == 'sin'):
        #env = HourlySimEnv(response='s',one_day=True, energy_in_state=False)
        env2 = HourlySimEnv(response='s', one_day=False, energy_in_state=energy, yesterday_in_state=False,
                            day_of_week = day_of_week)
    elif(response_type_str == 'mixed'):
        #env = HourlySimEnv(response='s',one_day=True, energy_in_state=False)
        env2 = HourlySimEnv(response='m', one_day=False, energy_in_state=energy, yesterday_in_state=False,
                            day_of_week = day_of_week)
    else:
        #env = HourlySimEnv(response='l',one_day=True, energy_in_state=False)
        env2 = HourlySimEnv(response='l', one_day=False, energy_in_state=energy,yesterday_in_state=False,
                            day_of_week = day_of_week)

    rewards = []
    rewards2 = []

    min_combined_losses = []
    min_policy_losses = []
    min_alpha_losses = []
    num_iters_list = []
    overall_best_action = None

    memory = ReplayMemory(replay_size)
    env = env2
    action_star = []
    state = None
    agent = SoftActorCritic(env.observation_space, env.action_space, memory)
    actions_2_save = []
    energy_usage = []
    start_flag = False
    while(env.day <= 60):
        step = env.day
        print("Day: " + str(step))
        print("Hour:")
        if(not start_flag):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            state = next_state
            start_flag = True
            continue

        if env.day <= 30:
            action = env.action_space.sample()  # Sample random action
            next_state, reward, done, info = env.step(action)
            #useless = env2.step(action)

            memory.push((state, action, reward, next_state, done))

            state = next_state
            continue
            
        else:
            if(memory.get_len() > batch_size):
                critic_1_losses = [None]
                critic_2_losses = [None]
                policy_losses = [None]
                alpha_losses = [None]
                actions = []
                for extra_step in range(extra_train):
                    print("--"*10)
                    print(" Extra Train " + str(extra_train))
                    q1_prev_loss = critic_1_losses[-1]
                    q2_prev_loss = critic_2_losses[-1]
                    policy_loss = policy_losses[-1]
                    alpha_loss = alpha_losses[-1]
                    return_update = agent.update_params(batch_size)
                    print(return_update)
                    print("--"*10)
                    critic_1_loss = return_update[0]
                    critic_2_loss = return_update[1]
                    policy_loss = return_update[2]
                    alpha_loss = return_update[3]

                    critic_1_losses.append(critic_1_loss)
                    critic_2_losses.append(critic_2_loss)
                    policy_losses.append(policy_loss)
                    alpha_losses.append(alpha_loss)
                    #num_iters_list.append(num_iters)
                    actions.append(agent.get_action(state))
            
            combined_q_loss = np.array(critic_1_losses[1:]) + np.array(critic_2_losses[1:])
            min_loss = np.amin(combined_q_loss)
            min_combined_losses.append(min_loss)
            index_of_min = np.where(combined_q_loss == min_loss)[0][0]
            action = actions[index_of_min] 
            # min_policy_losses.append(np.amin(np.array(policy_losses)))
            # min_alpha_losses.append(np.amin(np.array(alpha_losses)))

            next_state, reward, done, info = env.step(action)
            next_state = state
            energy_usage.append(env._simulate_humans(state, action)["avg"])

            memory.push((state, action, reward, next_state, done))
            
            #useless_next_state, reward2, useless_done, useless_info = env2.step(action)

            state = next_state
            action_star.append(action)
            if(done):
                rewards.append(reward)
                #rewards2.append(reward2)
                # actions_2_save.append(np.array(list.copy(action_star)))
                # action_star = []
            #rewards = [r[0] if r is np.ndarray else r for r in rewards]
            #rewards2 = [r[0] if r is np.ndarray else r for r in rewards2]
            print("--------" * 10)
    
    #return sum(rewards)
    # action_energy_pair = list(zip(actions_2_save,energy_usage))
    # df = pd.DataFrame(action_energy_pair, 
    #            columns =['Action Vector', 'Energy Used']) 
    # df.to_csv("action_energy_pair_" + response_type_str + "_extratrain_"+ str(extra_train) + ".csv")
    
    plt.figure()
    plt.plot(rewards, label='reward')
    plt.plot(moving_average(rewards),label='Moving Avg')
    plt.title("Rewards (Daily " + response_type_str + '| Total = ' + str(sum(rewards)), pad = 20.0)
    plt.legend()
    plt.xlabel("Day Number")
    plt.ylabel("Reward")
    plt.savefig(response_type_str +'_energy=' + str(energy) +'_training' + '_extratrain_' + str(extra_train) + '.png')

    # # # plt.figure()
    # # # plt.plot(rewards2, label='true reward')
    # # # plt.plot(moving_average(rewards2),label='Moving Avg')
    # # # plt.title("Daily Response (Trained on One_Day " + response_type_str + '| Total = ' + str(sum(rewards2)), pad = 20.0)
    # # # plt.legend()
    # # # plt.xlabel("Day Number")
    # # # plt.ylabel("Reward")
    # # # plt.savefig(response_type_str + '_results' + '_extratrain_' + str(extra_train) + '.png')

    plt.figure()
    plt.plot(min_combined_losses)
    plt.xlabel('Iteration ')
    plt.ylabel("Combined Q1+Q2 loss")
    plt.title("Combined Critic Loss at each hour", pad = 20.0)
    plt.savefig(response_type_str + '_energy=' + str(energy) + '_min_q_loss' + '_extratrain_' + str(extra_train) + '.png')

    return sum(rewards)

def train_curve_finder(max_iter):
    def train_store_rewards(response_type_str):
        rewards_list_e = []
        rewards_list_no_e = []
        for iteration in range(1,max_iter,10):
            #Add error bounds, just for loop then return avg, pointwise-max/min
            rewards_list_e.append(train(response_type_str,iteration, energy=True, day_of_week=True))
            rewards_list_no_e.append(train(response_type_str,iteration, energy=False, day_of_week=True))
        return np.array(rewards_list_e), np.array(rewards_list_no_e)
    # total_rewards_thresh_e, total_rewards_thresh_no_e = train_store_rewards('threshold_exp')
    # total_rewards_sin_e, total_rewards_sin_no_e  = train_store_rewards('sin')
    # total_rewards_linear_e, total_rewards_linear_no_e  = train_store_rewards('linear')
    total_rewards_mixed_e, total_rewards_mixed_no_e  = train_store_rewards('mixed')
    
    plt.figure()
    plt.plot(total_rewards_mixed_e, label='mixed-dow_-w/-energy', linestyle='dashed',color='#98ff98')
    plt.plot(total_rewards_mixed_no_e, label='mixed-dow_-w/o-energy',color='#98ff98')
    # plt.plot(total_rewards_thresh_e, label='threshold-exp-w/-energy', linestyle='dashed',color='#1B998B')
    # plt.plot(total_rewards_sin_e, label='sin-w/-energy', linestyle='dashed',color = '#ED217C')
    # plt.plot(total_rewards_linear_e, label='linear-w/-energy', linestyle='dashed',color = '#2D3047')

    # plt.plot(total_rewards_thresh_no_e, label='threshold-exp-w/o-energy', color='#1B998B')
    # plt.plot(total_rewards_sin_no_e, label='sin-w/o-energy', color = '#ED217C')
    # plt.plot(total_rewards_linear_no_e, label='linear-w/o-energy', color = '#2D3047')
    #plt.plot(moving_average(rewards),label='Moving Avg')
    plt.title("Total Rewards Learning Curve(H2H)", pad = 20.0)
    plt.legend()
    plt.xlabel("Number of Extra Train Iterations (Factor of 10")
    plt.ylabel("Total Reward")
    plt.savefig('total_reward_curve_mixed_h2h.png')



#Training Thresh-Exp
# train('threshold_exp',1)
# train('sin',1)
# train('linear',1)

# train('threshold_exp',10)
# train('sin',10)
# train('linear',10)

# train('threshold_exp',50)
# train('sin',50)
# train('linear',50)

# train('threshold_exp',100)
# train('sin',100)
# train('linear',100)

train_curve_finder(100)
