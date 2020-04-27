from Memory import ReplayMemory
from SACV2 import SoftActorCritic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
import behavioral_sim
from behavioral_sim.custom_envs import BehavSimEnv
from behavioral_sim.custom_envs import HourlySimEnv


"""
My rough code to train and experiment with SACV2, I pretty much just call the train function below


"""

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
    """
    Args: 
        Response_type_str = 'theshold_exp' or 'sin' or 'mixed' or 'linear'
        Extra_Train = Number of iterations to "overtrain"
        Energy: Whether or not to include previous day energy in the state
        Day_of_Week: Whether or not to include day_of_week multiplier
    
    Summary:
        This code 'simulates' a run of SACV2 training and acting over 30 days (takes a step each day)

    """
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
    elif(response_type_str == 'linear'):
        #env = HourlySimEnv(response='l',one_day=True, energy_in_state=False)
        env2 = HourlySimEnv(response='l', one_day=False, energy_in_state=energy,yesterday_in_state=False,
                            day_of_week = day_of_week)
    
    else:
        raise NotImplementedError

    #rewards in environment that agents see
    rewards = []

    #optional rewards list for environment that agent doesn't see (used in one-day training -> generalization case)
    rewards2 = []

    min_combined_losses = []
    min_policy_losses = []
    min_alpha_losses = []
    num_iters_list = []
    overall_best_action = None

    memory = ReplayMemory(replay_size)

    #Sometimes use 2 environment, this is used to default to 1 env. 
    #Change if you want to use 2 environments
    env = env2

    action_star = []
    state = None


    agent = SoftActorCritic(env.observation_space, env.action_space, memory)

    #Actions 2 save and energy_usage for data_generation
    # actions_2_save = []
    # energy_usage = []

    #Flag corresp to whether first state has been initialized
    start_flag = False

    while(env.day <= 60):
        step = env.day
        print("Day: " + str(step))
        print("Hour:" + str(env.hour))
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
            
            #Finds the action corresp to the lowest combined q-loss
            combined_q_loss = np.array(critic_1_losses[1:]) + np.array(critic_2_losses[1:])
            min_loss = np.amin(combined_q_loss)
            min_combined_losses.append(min_loss)
            index_of_min = np.where(combined_q_loss == min_loss)[0][0]
            action = actions[index_of_min] 

            # min_policy_losses.append(np.amin(np.array(policy_losses)))
            # min_alpha_losses.append(np.amin(np.array(alpha_losses)))

            next_state, reward, done, info = env.step(action)
            next_state = state

            memory.push((state, action, reward, next_state, done))
            
            #useless_next_state, reward2, useless_done, useless_info = env2.step(action)

            state = next_state

            #old code for saving data samples
            # actions_2_save.append(action[0])

            if(done):
                rewards.append(reward)
                #rewards2.append(reward2)

                #Old code for datageneration
                # env_usage = env._simulate_humans(state, action)["avg"]
                # for energy_i in env_usage:
                #     energy_usage.append(energy_i)
            
            #Old code when rewards where not fixed
            #rewards = [r[0] if r is np.ndarray else r for r in rewards]
            #rewards2 = [r[0] if r is np.ndarray else r for r in rewards2]
            print("--------" * 10)
    
    plt.figure()
    plt.plot(rewards, label='reward')
    plt.plot(moving_average(rewards),label='Moving Avg')
    plt.title("Rewards (H2H " + response_type_str + '| Total = ' + str(sum(rewards)), pad = 20.0)
    plt.legend()
    plt.xlabel("Day Number")
    plt.ylabel("Reward")
    plt.savefig(response_type_str +'_energy=' + str(energy) +'_training' + '_extratrain_' + str(extra_train) + '.png')

    # plt.figure()
    # plt.plot(rewards2, label='true reward')
    # plt.plot(moving_average(rewards2),label='Moving Avg')
    # plt.title("Daily Response (Trained on One_Day " + response_type_str + '| Total = ' + str(sum(rewards2)), pad = 20.0)
    # plt.legend()
    # plt.xlabel("Day Number")
    # plt.ylabel("Reward")
    # plt.savefig(response_type_str + '_results' + '_extratrain_' + str(extra_train) + '.png')

    plt.figure()
    plt.plot(min_combined_losses)
    plt.xlabel('Iteration ')
    plt.ylabel("Combined Q1+Q2 loss")
    plt.title("Combined Critic Loss at each hour", pad = 20.0)
    plt.savefig(response_type_str + '_energy=' + str(energy) + '_min_q_loss' + '_extratrain_' + str(extra_train) + '.png')

    return 
    # return sum(rewards)


    #Old code for generating datasamples
    # #return sum(rewards)
    # hours = [i % 10 for i in range(len(actions_2_save))]
    # df = pd.DataFrame(
    # {'Hour': hours,
    #  'Point': actions_2_save,
    #  'Energy': energy_usage
    # })
    # # df.to_csv("action_energy_" + response_type_str + "_extratrain_"+ str(extra_train-1) + ".csv")
    

# Function for training error bounds and total reward plots (very rough)
# def train_curve_finder(max_iter, response_type_str=None):
#     def train_store_rewards(response_type_str=None):
#         rewards_list_e = []
#         rewards_list_e_min = []
#         rewards_list_e_max = []
        
#         # rewards_list_no_e = []
#         # rewards_list_no_e_min = []
#         # rewards_list_no_e_max = []
#         for iteration in range(1,max_iter,20):
#             #Add error bounds, just for loop then return avg, pointwise-max/min
#             max_reward = -1e10
#             cum_reward = 0
#             min_reward = 0
#             for i in range(5):
#                 curr_total_rewards = train(response_type_str,iteration, energy=True, day_of_week=False)
#                 if(curr_total_rewards > max_reward):
#                     max_reward = curr_total_rewards
#                 if(curr_total_rewards < min_reward):
#                     min_reward = curr_total_rewards
#                 cum_reward += curr_total_rewards

#             rewards_list_e.append(cum_reward / 5)
#             rewards_list_e_min.append(min_reward)
#             rewards_list_e_max.append(max_reward)
#         return np.array(rewards_list_e), np.array(rewards_list_e_min), np.array(rewards_list_e_min)
    
#     rewards_thresh_e, min_rewards_thresh_e, max_rewards_thresh_e = train_store_rewards('threshold_exp')
#     rewards_sin, min_rewards_sin, max_rewards_sin = train_store_rewards('sin')
#     rewards_linear, min_rewards_linear, max_rewards_linear = train_store_rewards('linear')
    
#     # total_rewards_thresh_e, total_rewards_thresh_no_e = train_store_rewards('threshold_exp')
#     # total_rewards_sin_e, total_rewards_sin_no_e  = train_store_rewards('sin')
#     # total_rewards_linear_e, total_rewards_linear_no_e  = train_store_rewards('linear')
#     #total_rewards_mixed_e, total_rewards_mixed_no_e  = train_store_rewards(response_type_str)
    
#     plt.figure()
#     # plt.plot(total_rewards_mixed_e, label=response_type_str + '-dow_-w/-energy', linestyle='dashed',color='#ED217C')
#     # plt.plot(total_rewards_mixed_no_e, label=response_type_str + '-dow_-w/o-energy',color='#ED217C')

#     plt.plot(rewards_thresh_e, label='avg_threshold_exp',color='#1B998B')
#     plt.plot(rewards_sin, label='avg_sin',color = '#ED217C')
#     plt.plot(rewards_linear, label='avg_linear',color = '#2D3047')


#     plt.plot(min_rewards_thresh_e, label='min_thresh_exp', linestyle='dashed',color='#1B998B')
#     plt.plot(min_rewards_sin, label='min_sin',linestyle = 'dashed',color = '#ED217C')
#     plt.plot(min_rewards_linear, label='min_linear',linestyle='dashed', color = '#2D3047')

#     plt.plot(max_rewards_thresh_e, label='max_thresh_exp', linestyle='dashed',color='#1B998B')
#     plt.plot(max_rewards_sin, label='max_sin',linestyle = 'dashed',color = '#ED217C')
#     plt.plot(max_rewards_linear, label='max_linear',linestyle='dashed', color = '#2D3047')

#     # plt.plot(total_rewards_thresh_no_e, label='threshold-exp-w/o-energy', color='#1B998B')
#     # plt.plot(total_rewards_sin_no_e, label='sin-w/o-energy', color = '#ED217C')
#     # plt.plot(total_rewards_linear_no_e, label='linear-w/o-energy', color = '#2D3047')
#     #plt.plot(moving_average(rewards),label='Moving Avg')
   
#     plt.title("Total Rewards Curve", pad = 20.0)
#     plt.legend()
#     plt.xlabel("Number of Extra Train Iterations (Factor of 20")
#     plt.ylabel("Total Reward")
#     plt.savefig('total_reward_curve_w_error.png')



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

# train_curve_finder(100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SACV2 Run')

    #Response type
    parser.add_argument('--response_type', type=str, 
                        default='mixed',
                        choices=['mixed', 'threshold_exp', 'sin', 'linear'],
                        help='Number of extra-train iterations at each step')

    #Number of extra-train iterations
    parser.add_argument('--extra_train', type=int, 
                    default=1,
                    help='Number of extra-train iterations at each step')
    
    parser.add_argument('--energy_in_state', type=bool, 
                    default=True,
                    help='Boolean whether or not to include energy in state')

    #Day of Week Multiplier 
    parser.add_argument('--day_of_week', type=bool, 
                        default=True,
                        help='Boolean whether or not to include day-of-week multiplier')

    args = parser.parse_args()
    
    print(args, end="\n\n")

    train(response_type_str = args.response_type, 
        extra_train = args.extra_train, 
        energy = args.energy_in_state,
        day_of_week=args.day_of_week)
