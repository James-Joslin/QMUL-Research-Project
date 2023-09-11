import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

parent_dir = str(os.path.abspath(os.path.join(os.path.dirname( __file__ ))))#, os.pardir)))
data_files = glob(os.path.join(parent_dir, "*", "*.npz"), recursive=True)

methods = [data_string.split("/")[-1].split("_")[0] for data_string in data_files]
rewards = np.array([])
timings = np.array([])

for data in data_files:
    npz = np.load(data)
    datasets = npz.files
    if len(datasets) >= 2:
        # If rewards is empty, initialize it. Otherwise, hstack.
        print(len(datasets[0]))
        print(len(datasets[1]))
        if rewards.size == 0:
            rewards = npz[datasets[0]].reshape(-1,1)
        else:
            rewards = np.hstack((rewards, npz[datasets[0]].reshape(-1,1)))

        # If timings is empty, initialize it. Otherwise, hstack.
        if timings.size == 0:
            timings = npz[datasets[1]].reshape(-1,1)
        else:
            timings = np.hstack((timings, npz[datasets[1]].reshape(-1,1)))
    npz.close()  # Close the npz file after reading to free up resources

# Plot rewards
flier_properties = dict(marker='o', markerfacecolor='none', markersize=3, linestyle='none')
plt.figure(figsize=(6, 4), dpi = 150)
plt.boxplot(rewards, showmeans=True, meanline=True, meanprops=dict(color='red'), widths=(0.3,0.3), flierprops=flier_properties)  
plt.title('LunarLander-v2', y = -0.25)
plt.xlabel('Search Method')
plt.ylabel('Reward')
plt.xticks(ticks=range(1, rewards.shape[1] + 1), labels=methods)
plt.tight_layout()
plt.show()

# Plot timings
plt.figure(figsize=(6, 4), dpi = 150)
plt.boxplot(np.log(timings), showmeans=True, meanline=True, meanprops=dict(color='red'), widths=(0.3,0.3), flierprops=flier_properties)  
plt.title('LunarLander-v2', y = -0.25)
plt.xlabel('Search Method')
plt.ylabel(r'$\ln(\mathrm{Decision\ Timings})$')
plt.xticks(ticks=range(1, timings.shape[1] + 1), labels=methods)
plt.tight_layout()
plt.show()  