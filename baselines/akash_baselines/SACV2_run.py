from Memory import ReplayMemory
from SACV2 import SoftActorCritic
from custom_envs import BehavSimEnv
from custom_envs import HourlySimEnv
import matplotlib.pyplot as plt
import numpy as np

replay_size = 10000


total_numsteps = 60
start_steps = 30
batch_size = 5
action_star = None

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def train(response_type_str):
    if(response_type_str == 'threshxxold_exp'):
        env = HourlySimEnv(response='t', one_day=True)
        env2 = HourlySimEnv(response='t', one_day=False)
    elif(response_type_str == 'sin'):
        env = HourlySimEnv(response='s',one_day=True)
        env2 = HourlySimEnv(response='w', one_day=False)
    elif(response_type_str == 'mixed'):
        env = HourlySimEnv(response='m',one_day=True)
        env2 = HourlySimEnv(response='m', one_day=False)
    else:
        env = HourlySimEnv(response='l',one_day=True)
        env2 = HourlySimEnv(response='l', one_day=False)

    rewards = []
    rewards2 = []

    min_combined_losses = []
    min_policy_losses = []
    min_alpha_losses = []
    num_iters_list = []
    overall_best_action = None

    memory = ReplayMemory(replay_size)

    state = np.concatenate((env.prices[0], np.array([0]), np.array([0])))
    action_star = None
    for step in range(10*total_numsteps):
        print("Hour:")
        print(env.hour)
        print("\nStep: " + str(step) + " / " + str(10*total_numsteps))
        if step < start_steps:
            action = env.action_space.sample()  # Sample random action
            next_state, reward, done, info = env.step(action)
            useless = env2.step(action)

            memory.push((state, action, reward, next_state, done))

            state = next_state
            continue
            
        else:
            agent = SoftActorCritic(env.observation_space, env.action_space, memory)
            if(memory.get_len() > batch_size):
                critic_1_losses = [None]
                critic_2_losses = [None]
                policy_losses = [None]
                alpha_losses = [None]
                actions = []
                for extra_train in range(1):
                    print("--"*10)
                    print(" Extra Train " + str(extra_train))
                    q1_prev_loss = critic_1_losses[-1]
                    q2_prev_loss = critic_2_losses[-1]
                    policy_loss = policy_losses[-1]
                    alpha_loss = alpha_losses[-1]
                    return_update = agent.overfit_update_params(batch_size,q1_prev_loss,q2_prev_loss,policy_loss,alpha_loss)
                    print(return_update)
                    print("--"*10)
                    losses = return_update[0]
                    num_iters = return_update[1]

                    critic_1_loss = losses[0]
                    critic_2_loss = losses[1]
                    policy_loss = losses[2]
                    alpha_loss = losses[3]

                    critic_1_losses.append(critic_1_loss)
                    critic_2_losses.append(critic_2_loss)
                    policy_losses.append(policy_loss)
                    alpha_losses.append(alpha_loss)
                    num_iters_list.append(num_iters)
                    actions.append(agent.get_action(state))
            
            combined_q_loss = np.array(critic_1_losses[1:]) + np.array(critic_2_losses[1:])
            min_loss = np.amin(combined_q_loss)
            min_combined_losses.append(min_loss)
            index_of_min = np.where(combined_q_loss == min_loss)[0][0]
            action = actions[index_of_min] 
            # min_policy_losses.append(np.amin(np.array(policy_losses)))
            # min_alpha_losses.append(np.amin(np.array(alpha_losses)))

        next_state, reward, done, info = env.step(action)
        #next_state = state

        memory.push((state, action, reward, next_state, done))
        
        useless_next_state, reward2, useless_done, useless_info = env2.step(action)

        state = next_state
        action_star = action
        rewards.append(reward)
        rewards2.append(reward2)
        rewards = [r[0] if r is np.ndarray else r for r in rewards]
        rewards2 = [r[0] if r is np.ndarray else r for r in rewards2]
        print("--------" * 10)
    
    
    plt.figure()
    plt.plot(rewards, label='reward')
    plt.plot(moving_average(rewards),label='Moving Avg')
    plt.title("Rewards of new SAC V2 (Trained One-Day Hourly " + response_type_str, pad = 20.0)
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

    plt.figure()
    plt.plot(min_combined_losses)
    plt.xlabel('Iteration ')
    plt.ylabel("Combined Q1+Q2 loss")
    plt.title("Combined Critic Loss at each Day", pad = 20.0)
    plt.savefig(response_type_str + '_min_q_loss.png')




#Training Thresh-Exp
train('threshold_exp')
train('sin')
train('linear')
train('mixed')