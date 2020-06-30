import os
import matplotlib.pyplot as plt
import numpy as np

# sac_rewards_per_iter = np.load(os.getcwd() + "/runs_data/no_seed/SACV2_1_rewards_no_seed.npy")
# print(sac_rewards_per_iter)

def get_min_avg_max(id):
    runs = np.load(os.getcwd() + "/runs_data/seed/SACV2_" + str(id) + "_rewards_seed.npy")
    v1 = np.sum(runs[0])
    v2 = np.sum(runs[1])
    v3 = np.sum(runs[2])
    v4 = np.sum(runs[3])
    v5 = np.sum(runs[4])
    
    min_reward = min(v1,v2,v3,v4,v5)
    avg_reward = (v1+v2+v3+v4+v5) / 5
    max_reward = max(v1,v2,v3,v4,v5)
    return min_reward, avg_reward, max_reward

min_runs = []
avg_runs = []
max_runs = []
for i in range(1,101,10):
    min_run, avg_run, max_run = get_min_avg_max(i)
    min_runs.append(min_run)
    avg_runs.append(avg_run)
    max_runs.append(max_run)

plt.figure()
plt.plot([0,10,20,30,40,50,60,70,80,90],min_runs, linestyle = 'dashed', label='Min Reward')
plt.plot([0,10,20,30,40,50,60,70,80,90],avg_runs, label='Avg Reward')
plt.plot([0,10,20,30,40,50,60,70,80,90],max_runs, linestyle='dashed', label = 'Max Reward')
plt.xlabel("Extra Train Iterations")
plt.ylabel("Total Reward")
plt.legend()
plt.title("Effects of Extra Training on Cumulative Reward (Mixed Response)")
plt.savefig("all_seed.png", dpi=100)
    
