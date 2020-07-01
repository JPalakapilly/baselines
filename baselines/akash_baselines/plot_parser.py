import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sac_rewards_per_iter = np.load(os.getcwd() + "/runs_data/no_seed/SACV2_1_rewards_no_seed.npy")
# print(sac_rewards_per_iter)

# def get_min_avg_max(id, response):
#     runs = np.load(os.getcwd() + "/runs_data/no_seed/" + response +  
#                         "/SACV2_" + response + "_" + str(id) + "_rewards_no_seed.npy")
#     v1 = np.sum(runs[0])
#     v2 = np.sum(runs[1])
#     v3 = np.sum(runs[2])
#     v4 = np.sum(runs[3])
#     v5 = np.sum(runs[4])
    
#     min_reward = min(v1,v2,v3,v4,v5)
#     avg_reward = (v1+v2+v3+v4+v5) / 5
#     max_reward = max(v1,v2,v3,v4,v5)
#     return min_reward, avg_reward, max_reward

# min_runs_m = []
# avg_runs_m = []
# max_runs_m = []
# min_runs_t = []
# avg_runs_t = []
# max_runs_t = []
# min_runs_l = []
# avg_runs_l = []
# max_runs_l = []
# for i in range(1,101,10):
#     min_run_m, avg_run_m, max_run_m = get_min_avg_max(i,'mixed')
#     # min_run_l, avg_run_l, max_run_l = get_min_avg_max(i,'linear')
#     # min_run_t, avg_run_t, max_run_t = get_min_avg_max(i,'threshold_exp')
#     min_runs_m.append(min_run_m)
#     avg_runs_m.append(avg_run_m)
#     max_runs_m.append(max_run_m)

#     # min_runs_l.append(min_run_l)
#     # avg_runs_l.append(avg_run_l)
#     # max_runs_l.append(max_run_l)

#     # min_runs_t.append(min_run_t)
#     # avg_runs_t.append(avg_run_t)
#     # max_runs_t.append(max_run_t)

# plt.figure()
# plt.plot([0,10,20,30,40,50,60,70,80,90],min_runs_m, color='orange', linestyle = 'dashed')
# plt.plot([0,10,20,30,40,50,60,70,80,90],avg_runs_m, color='orange',label='Mixed')
# plt.plot([0,10,20,30,40,50,60,70,80,90],max_runs_m, color='orange', linestyle='dashed')

# # plt.plot([0,10,20,30,40,50,60,70,80,90],min_runs_l, color='blue', linestyle = 'dashed')
# # plt.plot([0,10,20,30,40,50,60,70,80,90],avg_runs_l, color='blue',label='Linear')
# # plt.plot([0,10,20,30,40,50,60,70,80,90],max_runs_l, color='blue', linestyle='dashed')

# # plt.plot([0,10,20,30,40,50,60,70,80,90],min_runs_t, color='purple', linestyle = 'dashed')
# # plt.plot([0,10,20,30,40,50,60,70,80,90],avg_runs_t, color='purple',label='Thresh-exp')
# # plt.plot([0,10,20,30,40,50,60,70,80,90],max_runs_t, color='purple', linestyle='dashed')

# plt.xlabel("Extra Train Iterations")
# plt.ylabel("Total Reward")
# plt.legend()
# plt.title("Effects of Extra Training on Cumulative Reward")
# plt.savefig(os.getcwd() + "/runs_data/no_seed/mixed/effects_m.png", dpi=100)
    

runs_data = pd.read_csv("threshold_exp_extratrain_0.csv")
first_30_actions = np.array(runs_data['Point'][0:300])
actions_sac_star = []
for i in range(0,300,10):
    curr_action = first_30_actions[i:i+10]
    actions_sac_star.append(curr_action)

actions_sac_min = np.amin(actions_sac_star,axis=0)
actions_sac_max = np.amax(actions_sac_star,axis=0)


actions_star = np.load("action_star_bcql.npy")
actions_min = np.amin(actions_star,axis=0)
actions_max = np.amax(actions_star,axis=0)
# print(actions_max)
action_bcql_average = np.zeros_like(actions_star[0])
action_sac_average = np.zeros_like(actions_sac_star[0])
for i in range(30):
    action_bcql_average += actions_star[i]
    action_sac_average += actions_sac_star[i]
action_bcql_average = action_bcql_average / 30
action_sac_average = action_sac_average / 30

plt.figure()
plt.xlabel("Hour (8AM-5PM)")
plt.ylabel("Points")
plt.plot([8,9,10,11,12,13,14,15,16,17],actions_min,linestyle='dashed',color='#2a9d8f')
plt.plot([8,9,10,11,12,13,14,15,16,17],action_bcql_average,linestyle='solid',color='#2a9d8f', label = 'BCQL Average Point Value')
plt.plot([8,9,10,11,12,13,14,15,16,17],actions_max,linestyle='dashed',color='#2a9d8f')
# plt.plot([8,9,10,11,12,13,14,15,16,17],actions_sac_min,linestyle='dashed',color='#f4a261')
# plt.plot([8,9,10,11,12,13,14,15,16,17],action_sac_average,linestyle='solid',color='#f4a261', label = 'SAC Average Point Value')
# plt.plot([8,9,10,11,12,13,14,15,16,17],actions_sac_max,linestyle='dashed',color='#f4a261')
# plt.legend(loc='lower right')
#plt.title("BCQL and SAC Exploration")
# plt.title("SAC's Varied Exploration")
plt.title("BCQL's Restricted Exploration")
# plt.plot([i+1 for i in range(0,30)],actions_init,'r-')
plt.savefig("bcql_action_explore.png")