from Memory import ReplayMemory
from SACV2 import SoftActorCritic
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys
sys.path.append("..")
from behavioral_sim.custom_envs import BehavSimEnv
from behavioral_sim.custom_envs import HourlySimEnv
import IPython
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import pickle


"""
My rough code to train and experiment with SACV2, I pretty much just call the train function below


"""

replay_size = 10000


total_numsteps = 60
start_steps = 30
batch_size = 1
action_star = None

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def load_model_from_disk(file_name = "GPyOpt_planning_model"):
    json_file = open(file_name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(file_name + ".h5")
    print("Loaded model from disk")

    loaded_model.compile(loss="mse", optimizer="adam")
    return loaded_model

def train_planning_model(
        energy_today, 
        action, 
        day_of_week, 
        loaded_model, 
        filename_to_save = "GPyOpt_planning_model"
    ):
        
    ## load the minMaxScalers
    with open ("scaler_X.pickle", "rb") as input_file:
        scaler_X = pickle.load(input_file) 
    with open ("scaler_y.pickle", "rb") as input_file:
        scaler_y = pickle.load(input_file) 

    ## prepare the data

    d_X = pd.DataFrame(data = { "action" : action, "dow" : day_of_week } )
    scaled_X = scaler_X.transform(d_X)
    sxr = scaled_X.reshape((scaled_X.shape[0], 1, scaled_X.shape[1])) 
    
    d_y = pd.DataFrame(data = {"energy" : energy_today})
    scaled_y = scaler_y.transform(d_y)
    
    loaded_model.fit(
            sxr,    ## these all need to be changed if the GPyOpt evaluates differently
            scaled_y,
            epochs=100,
            batch_size=10,
            validation_split=0.0,
            verbose=0,
        )
    
    model_json = loaded_model.to_json()
    with open(filename_to_save + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    loaded_model.save_weights(filename_to_save + ".h5")
    print("Saved model to disk")
    
    return

def planning_prediction(action, day_of_week, loaded_model):

    ## load the minMaxScalers
    with open ("scaler_X.pickle", "rb") as input_file:
        scaler_X = pickle.load(input_file) 
    with open ("scaler_y.pickle", "rb") as input_file:
        scaler_y = pickle.load(input_file) 

    ## prepare the data

    d_X = pd.DataFrame(data = { "action"  : action, "dow" : day_of_week } )
    scaled_X = scaler_X.transform(d_X)
    sxr = scaled_X.reshape((scaled_X.shape[0], 1, scaled_X.shape[1])) 

    preds = loaded_model.predict(sxr)

    inv_preds = scaler_y.inverse_transform(preds)  

    return np.squeeze(inv_preds)

def series_to_supervised(data,
                         n_in,
                         target_features,
                         col_names=None,
                         n_out=1,
                         dropnan=True,
                         initial_excluded_timesteps=0,
                         removed_features=[],
                         only_current_timestep_features=[]):
    """Takes time series data and converts it into a supervised learning
    problem framework.

        Parameters:
            - data (pd.Dataframe) -- the time series data to be converted.
            - n_in (int) -- Number of time steps to use as lag for the feature 
                matrix
            - col_names (List[str]) -- list of strings to use as column names,
                that get converted into features for each time lag
            - target_features (List[str]) -- List of features that will be used
                as dependent variables in the target matrix.
            - n_out (int) -- Number of time steps to use as lag for the target
                matrix
            - dropnan (bool) -- Whether to drop nan values
            - initial_excluded_timesteps (int) -- The number of input timesteps to 
                ignore before starting the time lag.
            - removed_features (List[str]) -- List of features that should be removed
                from the dataframe. 
            - only_current_timestep_features (List[str]) -- Features that should
            only be included in the current timestep, not any before (e.g.) to avoid
            unintended dependencies
            
            - [Planned] exclude_current_day (bool) -- Whether to include values 
                from the current day. If this parameter is false, then the time lag 
                will always start with the day preceding the current time 
                step.
                
        Outputs:
            - (X, y): (Feature matrix, target matrix)
    """
    
    if col_names is None:
        col_names = data.columns
        
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data).drop(removed_features, axis=1)
    col_names = [x for x in col_names if x not in removed_features]
    cols, names = list(), list()      
        
    only_prev_time_features = df.drop(only_current_timestep_features, axis=1)
    only_prev_column_names = [x for x in col_names if x in only_prev_time_features.columns]

    # (t-n, ... t-1) --> i.e. steps into the past
    for i in range(n_in + initial_excluded_timesteps, initial_excluded_timesteps, -1):
        cols.append(only_prev_time_features.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in only_prev_column_names]

    # (t, t+1, ... t+n) --> i.e. steps into the future
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (col)) for col in col_names]
        else:
            names += [('%s(t + %d)' % (col, i)) for col in col_names]

    # concat
    agg = pd.concat(cols, axis=1)
#     agg.columns = names

    # dropnan
    if dropnan:
        agg.dropna(inplace=True)
    
    if target_features: 
        Y_vals_cols = ([('%s(t)' % (col)) for col in target_features] + 
            ['%s(t + %d)' % (col, i) for col in target_features for i in range(1, n_out)])

        Y_vals = agg[Y_vals_cols]
        X_vals = agg.drop(Y_vals_cols, axis=1)

        agg.columns = names
              
        return X_vals, Y_vals # X_vals.values, Y_vals.values
    
    else:
        return agg, _

def planning_prediction(action, day_of_week): # player name


    ## load the minMaxScalers
    with open ("scaler_X.pickle", "rb") as input_file:
        scaler_X = pickle.load(input_file) 
    with open ("scaler_y.pickle", "rb") as input_file:
        scaler_y = pickle.load(input_file) 

    ## prepare the data

    d_X = pd.DataFrame(data = { "action"  : action, "dow" : day_of_week } )
    scaled_X = scaler_X.transform(d_X)
    sxr = scaled_X.reshape((scaled_X.shape[0], 1, scaled_X.shape[1])) 

    # supervised_X, _ = series_to_supervised(data = scaled_X, 
    #                         target_features = None, # target_features = ['Energy'], 
    #                         only_current_timestep_features=[],
    #                         initial_excluded_timesteps=10,
    #                         col_names = ["Point", "Day of Week"], # df.columns.tolist(), 
    #                         n_in = 40,
    #                         n_out = 1)

    ## load the trained nn
    json_file = open("GPyOpt_planning_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("GPyOpt_planning_model.h5")
    print("Loaded model from disk")

    loaded_model.compile(loss="mse", optimizer="adam")

    preds = loaded_model.predict(sxr)

    inv_preds = scaler_y.inverse_transform(preds)  

    return np.squeeze(inv_preds)


def train(response_type_str, 
    extra_train, 
    planning_iterations,
    one_day = False, 
    energy=True, 
    day_of_week=True, 
    planning_model = False,
    train_planning_model_with_new_data = False,
    ):
    """
    Args: 
        Response_type_str = 'theshold_exp' or 'sin' or 'mixed' or 'linear'
        Extra_Train = Number of iterations to "overtrain"
        planning_iterations = number of times to query the planning model
        One_day: Whether to train from a single day's price signal 
        Energy: Whether or not to include previous day energy in the state
        Day_of_Week: Whether or not to include day_of_week multiplier
        planning_model: whether or not to use the planning model
        train_planning_model_with_new_data= whether to train the planning model
    
    Summary:
        This code 'simulates' a run of SACV2 training and acting over 30 days (takes a step each day)

    """
    if(response_type_str == 'threshold_exp'):
        #env = HourlySimEnv(response='t', one_day=True, energy_in_state=False)
        env2 = BehavSimEnv(response='t', one_day = one_day, energy_in_state=energy, yesterday_in_state=False,
                            day_of_week = day_of_week)
    elif(response_type_str == 'sin'):
        #env = HourlySimEnv(response='s',one_day=True, energy_in_state=False)
        env2 = BehavSimEnv(response='s', one_day=one_day, energy_in_state=energy, yesterday_in_state=False,
                            day_of_week = day_of_week)
    elif(response_type_str == 'mixed'):
        #env = HourlySimEnv(response='s',one_day=True, energy_in_state=False)
        env2 = BehavSimEnv(response='m', one_day=one_day, energy_in_state=energy, yesterday_in_state=False,
                            day_of_week = day_of_week)
    elif(response_type_str == 'linear'):
        #env = HourlySimEnv(response='l',one_day=True, energy_in_state=False)
        env2 = BehavSimEnv(response='l', one_day=one_day, energy_in_state=energy,yesterday_in_state=False,
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

    reward_planning = []
    
    # Actions 2 save and energy_usage for data_generation
    # actions_2_save = []
    # energy_usage = []

    # Flag corresp to whether first state has been initialized
    start_flag = False

    while (env.day <= 60):
        step = env.day
        day_of_week = env.day % 7 
        print("Day: " + str(step))
        if (not start_flag):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            state = np.copy(next_state)
            start_flag = True
            continue

        if env.day <= 30:
            action = env.action_space.sample()  # Sample random action
            next_state, reward, done, info = env.step(action)

            memory.push((state, action, reward, next_state, done))

            state = np.copy(next_state)
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
                    print(" Extra Train " + str(extra_step))
                    q1_prev_loss = critic_1_losses[-1]
                    q2_prev_loss = critic_2_losses[-1]
                    policy_loss = policy_losses[-1]
                    alpha_loss = alpha_losses[-1]
                    return_update = agent.update_params(batch_size, memory_type = "replay")
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

                energy_yesterday = np.copy(next_state[:10])

                if planning_model:
                    
                    ## load model from disk now   
                    if train_planning_model_with_new_data:
                        if (env.day==31):
                            loaded_model = load_model_from_disk("GPyOpt_planning_model")
                        else:
                            loaded_model = load_model_from_disk("GPyOpt_planning_model_training")
                    else: 
                        loaded_model = load_model_from_disk("GPyOpt_planning_model")
            
                    for planning_step in range(planning_iterations):
                        print("--"*10)
                        print(" planning step " + str(planning_step))
                        q1_prev_loss = critic_1_losses[-1]
                        q2_prev_loss = critic_2_losses[-1]
                        policy_loss = policy_losses[-1]
                        alpha_loss = alpha_losses[-1]

                        grid_prices_today = env.prices[env.day] # assuming that the env.day here correctly is set to the correct day
                        grid_prices_tmr = env.prices[(env.day + 1) % 365]
                        # state is defined as [previous day energy, current day prices] 
                        # I don't currently want this to change, as it should be the same to allow agent to explore

                        state = np.concatenate((energy_yesterday, grid_prices_today))
                        action = agent.get_action(state)
                     
                        planned_energy_consumption = planning_prediction(action, day_of_week, loaded_model)
                        
                        # will define next state as [energy, grid prices tomorrow]

                        next_state = np.concatenate((planned_energy_consumption, grid_prices_tmr))
                        reward = env.get_reward_planning_model(grid_prices_today, planned_energy_consumption) 
                        done = True

                        agent.planning_replay_memory.push((state, action, reward, next_state, done))
                        
                        return_update = agent.update_params(batch_size, memory_type = "planning")

#                         IPython.embed()
                        
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
                        reward_planning.append(reward)
                        # num_iters_list.append(num_iters)
                        # actions.append(agent.get_action(state))
            
            #Finds the action corresp to the lowest combined q-loss
            # combined_q_loss = np.array(critic_1_losses[1:]) + np.array(critic_2_losses[1:])
            # min_loss = np.amin(combined_q_loss)
            # min_combined_losses.append(min_loss)
            # index_of_min = np.where(combined_q_loss == min_loss)[0][0]
            
            action = agent.get_action(state)

            # min_policy_losses.append(np.amin(np.array(policy_losses)))
            # min_alpha_losses.append(np.amin(np.array(alpha_losses)))

            next_state, reward, done, info = env.step(action)

            memory.push((state, action, reward, next_state, done))
            
            # train the planning model 
            energy_today = np.copy(next_state[:10])
                        
            # if training the planning model is the flag that we'll change, then 
            # change over to "planning_model_training" and train it 
            
            if train_planning_model_with_new_data:
                if env.day == 31:
                    loaded_model = load_model_from_disk("GPyOpt_planning_model_training")
                    
                train_planning_model(    
                    energy_today = energy_today, 
                    action = action, 
                    day_of_week = day_of_week, 
                    loaded_model = loaded_model
                )
            
            state = np.copy(next_state)

            if(done):
                rewards.append(reward)
            print("--------" * 10)
    
    return rewards, critic_1_losses, critic_2_losses, policy_losses, alpha_losses, reward_planning




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
def train_curve_finder(max_iter, response_type_str):
    def train_store_rewards(response_type_str):
        sampled_days = [19,16,29,18,14,23,9,21,10,30]
        #Key = Day | Val = list for SAC Reward
        rewards_dict = {i: [] for i in range(1,max_iter,10)}
        for iteration in range(1,max_iter,10):
            #Add error bounds, just for loop then return avg, pointwise-max/min
            for i in range(5):
                curr_rewards_sac = train(response_type_str,iteration, energy=True, day_of_week=True)
                rewards_dict[iteration].append(curr_rewards_sac)
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
        
    sac_rewards_dict = train_store_rewards(response_type_str)
    for et in sac_rewards_dict.keys():
        sac_rewards_iter_et = sac_rewards_dict[et]
        print(sac_rewards_iter_et)
        print("----\n"*2)
        np.save("SACV2_" + response_type_str + "_" + str(et) + "_rewards_no_seed",sac_rewards_iter_et)


    # rewards_thresh_e, min_rewards_thresh_e, max_rewards_thresh_e = train_store_rewards('threshold_exp')
    # rewards_sin, min_rewards_sin, max_rewards_sin = train_store_rewards('sin')
    # rewards_linear, min_rewards_linear, max_rewards_linear = train_store_rewards('linear')
    
    # total_rewards_thresh_e, total_rewards_thresh_no_e = train_store_rewards('threshold_exp')
    # total_rewards_sin_e, total_rewards_sin_no_e  = train_store_rewards('sin')
    # total_rewards_linear_e, total_rewards_linear_no_e  = train_store_rewards('linear')
    #total_rewards_mixed_e, total_rewards_mixed_no_e  = train_store_rewards(response_type_str)
    
    # plt.figure()
    # plt.plot(total_rewards_mixed_e, label=response_type_str + '-dow_-w/-energy', linestyle='dashed',color='#ED217C')
    # plt.plot(total_rewards_mixed_no_e, label=response_type_str + '-dow_-w/o-energy',color='#ED217C')

    # plt.plot(rewards_thresh_e, label='avg_threshold_exp',color='#1B998B')
    # plt.plot(rewards_sin, label='avg_sin',color = '#ED217C')
    # plt.plot(rewards_linear, label='avg_linear',color = '#2D3047')


    # plt.plot(min_rewards_thresh_e, label='min_thresh_exp', linestyle='dashed',color='#1B998B')
    # plt.plot(min_rewards_sin, label='min_sin',linestyle = 'dashed',color = '#ED217C')
    # plt.plot(min_rewards_linear, label='min_linear',linestyle='dashed', color = '#2D3047')

    # plt.plot(max_rewards_thresh_e, label='max_thresh_exp', linestyle='dashed',color='#1B998B')
    # plt.plot(max_rewards_sin, label='max_sin',linestyle = 'dashed',color = '#ED217C')
    # plt.plot(max_rewards_linear, label='max_linear',linestyle='dashed', color = '#2D3047')

    # plt.plot(total_rewards_thresh_no_e, label='threshold-exp-w/o-energy', color='#1B998B')
    # plt.plot(total_rewards_sin_no_e, label='sin-w/o-energy', color = '#ED217C')
    # plt.plot(total_rewards_linear_no_e, label='linear-w/o-energy', color = '#2D3047')
    #plt.plot(moving_average(rewards),label='Moving Avg')
   
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
                    default=101,
                    help='Number of extra-train iterations at each step')
    
    parser.add_argument('--energy_in_state', type=bool, 
                    default=True,
                    help='Boolean whether or not to include energy in state')

    #Day of Week Multiplier 
    parser.add_argument('--day_of_week', type=bool, 
                        default=True,
                        help='Boolean whether or not to include day-of-week multiplier')

    parser.add_argument("--planning_model", type = bool,
                        default = False,
                        help = "Boolean whether or not to include planning model")

    parser.add_argument("--planning_iterations", type = int,
                        default = 10,
                        help = "number of times to explore training model")


    args = parser.parse_args()
    
    print(args, end="\n\n")

    train(response_type_str = args.response_type, 
         extra_train = args.extra_train, 
         energy = args.energy_in_state,
         day_of_week = args.day_of_week,
         planning_model = args.planning_model,
         planning_iterations = args.planning_iterations
         )
    #train_curve_finder(args.extra_train,args.response_type)
