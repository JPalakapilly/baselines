from Memory import ReplayMemory
from SACV2 import SoftActorCritic
from custom_envs import BehavSimEnv
import matplotlib.pyplot as plt
import numpy as np

replay_size = 10000


total_numsteps = 60
start_steps = 30
batch_size = 10
action_star = None

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def train(response_type_str):
    if(response_type_str == 'threshold_exp'):
        env = BehavSimEnv()
    elif(response_type_str == 'sin'):
        env = BehavSimEnv('s')
    else:
        env = BehavSimEnv('l')

    rewards = []

    min_combined_losses = []
    min_policy_losses = []
    min_alpha_losses = []
    num_iters_list = []

    memory = ReplayMemory(replay_size)
    print(env.action_space)
    agent = SoftActorCritic(env.observation_space, env.action_space, memory)

    state = env.prices[0]
    for step in range(total_numsteps):
        print("\nStep: " + str(step) + " / " + str(total_numsteps))

        if step < start_steps:
            action = env.action_space.sample()  # Sample random action
            next_state, reward, done, info = env.step(action)
        
            memory.push((state, action, reward, next_state, done))

            action_star = action
            continue
            
        else:
            action_star = None
            min_combined_loss = 1e10
            if(memory.get_len() > batch_size):
                # critic_1_losses = []
                # critic_2_losses = []
                # policy_losses = []
                # alpha_losses = []
                for training_iter in range(1000):
                    print("Training Iter: " + str(training_iter))
                    q1_loss, q2_loss, policy_loss, alpha_loss = agent.update_params(batch_size)
                    # critic_1_losses.append(q1_loss)
                    # critic_2_losses.append(q2_loss)
                    # policy_losses.append(policy_loss)
                    # alpha_losses.append(alpha_loss)

                    combined_q_loss = q1_loss + q2_loss
                    if(combined_q_loss < min_combined_loss):
                        action_star = agent.get_action(state)
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

        memory.push((state, action, reward, next_state, done))

        state = next_state
        action_star = action
        rewards.append(reward)
        rewards = [r[0] if r is np.ndarray else r for r in rewards]
        print("--------" * 10)
    
    plt.figure(0)
    plt.plot(rewards, label='normal reward')
    plt.plot(moving_average(rewards),label='Moving Avg')
    plt.title("Rewards of SAC V2 (Trained on Year/Overfitting Idea) " + response_type_str)
    plt.legend()
    plt.xlabel("Day Number")
    plt.ylabel("Reward")
    plt.savefig(response_type_str + '_training.png')

    plt.figure(1)
    plt.plot(min_combined_losses)
    plt.xlabel('Day of the Month (1000 Training iters, used best action')
    plt.ylabel("Combined Q1+Q2 loss")
    plt.title("Min Combined Critic Loss at each Day")
    plt.savefig(response_type_str + '_min_q_loss.png')




#Training Thresh-Exp
train('threshold_exp')