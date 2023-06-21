import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import signal
import scipy.io
import argparse

plt.rcParams.update({'font.size': 14})

def main():

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--plot', type=str, default="kinematics",
                        help='kinematics, sim_x, sim_y, med')

    args = parser.parse_args()

    data_1_kinematics = np.load('mouse_experiments/data/mouse_1.npy')
    data_fast_kinematics = np.load('mouse_experiments/data/mouse_fast.npy')
    data_slow_kinematics = np.load('mouse_experiments/data/mouse_slow.npy')

    ##################### DATA FAST ############################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt_fast.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_fast = data[231:401:1] * -1
    data_fast = [-13.45250312, *data_fast[8:]]

    #################### DATA SLOW #########################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt_slow.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_slow = data[256:476:1] * -1

    #################### DATA_1 #########################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt1.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_1= data[226:406:1] * -1
    data_1 = [-13.45250312, *data_1[4:]]

    scale = 21
    offset = -.71

    data_fast = np.array(data_fast) / scale - offset
    data_slow = np.array(data_slow) / scale - offset
    data_1 = np.array(data_1) / scale - offset

    fast_mse = (np.sum((data_fast - data_fast_kinematics)**2)) / 163
    slow_mse = (np.sum((data_slow - data_slow_kinematics)**2)) / 220
    med_mse = (np.sum((data_1 - data_1_kinematics)**2)) / 177

    print('Difference from target trajectory fast (MSE): {}'.format(fast_mse))
    print('Difference from target trajectory slow (MSE): {}'.format(slow_mse))
    print('Difference from target trajectory med (MSE): {}'.format(med_mse))

    # Plot the kinematics (currently for simulated)
    plt.plot(data_fast + .02, label='Experimental', linewidth=4, color='orange')
    plt.plot(data_fast_kinematics + .02, label='Model Output', linewidth=4, linestyle='dashed', color='black')

    plt.plot(data_1, linewidth=4, color='orange')
    plt.plot(data_1_kinematics, linewidth=4, color='black', linestyle='dashed')

    plt.plot(data_slow - .02, linewidth=4, color='orange')
    plt.plot(data_slow_kinematics - .02, linewidth=4, linestyle='dashed', color='black')

    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.yticks([])
    plt.savefig('mouse_experiments/data/data_all_kinematics_plot.png')
    plt.show()

if __name__ == '__main__':
    main()