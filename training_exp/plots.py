import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

policy_losses = np.loadtxt('policy_losses.txt')
rewards = np.loadtxt('rewards_sim.txt')

plt.plot(rewards)
plt.ylabel('Reward', fontsize='14')
plt.xlabel('Iteration', fontsize='14')
plt.savefig('rewards_sim.png')