from Memory import ReplayMemory
from SACV2 import SoftActorCritic
from custom_envs import BehavSimEnv
import matplotlib.pyplot as plt
import numpy as np

replay_size = 10000


total_numsteps = 60
start_steps = 30
batch_size = 20
action_star = None

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def train(response_type_str):
    if(response_type_str == 'threshold_exp'):
        env2 = BehavSimEnv(response='t')
        env = BehavSimEnv(one_day = True, response='t')
    elif(response_type_str == 'sin'):
        env2 = BehavSimEnv(response='s')
        env = BehavSimEnv(one_day = True, response='s')
    else:
        env2 = BehavSimEnv(response='l')
        env = BehavSimEnv(one_day = True, response='l')

    rewards = []
    rewards2 = []

    min_combined_losses = []
    min_policy_losses = []
    min_alpha_losses = []
    num_iters_list = []
    overall_best_action = None

    memory = ReplayMemory(replay_size)
    agent = SoftActorCritic(env.observation_space, env.action_space, memory)

    energy = [  0.27906955, 11.89568229, 16.33842439, 16.79616623,  17.43101761,16.15182342,  16.23424318, 15.88182418,  15.08545289, 35.60408169, 123.49742958, 148.69794274, 158.48681169, 149.13342321, 159.31826339, 157.61794021, 158.80197216, 156.49390761, 147.03826373,  70.76004005, 42.86648745,  23.13229363,  22.51826147,  16.79616935]
    state = env.prices[0]
    action_star = None
    for step in range(total_numsteps):
        print("\nStep: " + str(step) + " / " + str(total_numsteps))
        if step < start_steps:
            action = env.action_space.sample()  # Sample random action
            next_state, reward, done, info = env.step(action)
            useless = env2.step(action)
            next_state = state

            memory.push((state, action, reward, next_state, done))

            state = next_state
            continue
            
        else:
            min_combined_loss = 1e10
            if(memory.get_len() > batch_size):
                # critic_1_losses = []
                # critic_2_losses = []
                # policy_losses = []
                # alpha_losses = []
                for training_iter in range(2000):
                    print("Step: " + str(step) + " Training Iter: " + str(training_iter))
                    q1_loss, q2_loss, policy_loss, alpha_loss = agent.update_params(batch_size)
                    # critic_1_losses.append(q1_loss)
                    # critic_2_losses.append(q2_loss)
                    # policy_losses.append(policy_loss)
                    # alpha_losses.append(alpha_loss)

                    combined_q_loss = q1_loss + q2_loss
                    if(combined_q_loss < min_combined_loss):
                        curr_action = agent.get_action(state)
                        if(not np.any(np.isnan(curr_action))):
                            action_star = curr_action
                            min_combined_loss = combined_q_loss
            #didn't know how to find min loss b/c of clipping
            # combined_q_loss = np.array(critic_1_losses) + np.array(critic_2_losses)
            # min_loss = np.amin(combined_q_loss)
            # index_of_min = np.where(combined_q_loss == min_loss)[0][0]
            # action = actions[index_of_min]
            action = action_star
            
            min_combined_losses.append(min_combined_loss)
            # min_policy_losses.append(np.amin(np.array(policy_losses)))
            # min_alpha_losses.append(np.amin(np.array(alpha_losses)))

        next_state, reward, done, info = env.step(action)
        next_state = state

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
    plt.plot(rewards, label='normal reward')
    plt.plot(moving_average(rewards),label='Moving Avg')
    plt.title("Rewards of SAC V2 (Trained on One_Day " + response_type_str)
    plt.legend()
    plt.xlabel("Day Number")
    plt.ylabel("Reward")
    plt.savefig(response_type_str + '_training.png')

    plt.figure()
    plt.plot(rewards2, label='normal reward')
    plt.plot(moving_average(rewards),label='Moving Avg')
    plt.title("Daily Response (Trained on One_Day " + response_type_str)
    plt.legend()
    plt.xlabel("Day Number")
    plt.ylabel("Reward")
    plt.savefig(response_type_str + '_results.png')

    plt.figure()
    plt.plot(min_combined_losses)
    plt.xlabel('Day of the Month (1000 Training iters, used best action')
    plt.ylabel("Combined Q1+Q2 loss")
    plt.title("Min Combined Critic Loss at each Day")
    plt.savefig(response_type_str + '_min_q_loss.png')




#Training Thresh-Exp
train('threshold_exp')
train('sin')
train('linear')