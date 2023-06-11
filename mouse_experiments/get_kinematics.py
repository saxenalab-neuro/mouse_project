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

    #b_1, a_1 = signal.butter(10, 20, fs=177, btype='lowpass')
    #b_fast, a_fast = signal.butter(10, 20, fs=163, btype='lowpass')
    #b_slow, a_slow= signal.butter(10, 20, fs=220, btype='lowpass')

    data_1_kinematics = np.loadtxt('scripts/mouse_1.txt')
    data_fast_kinematics = np.loadtxt('scripts/mouse_fast.txt')
    data_slow_kinematics = np.loadtxt('scripts/mouse_slow.txt')

    #medium_kinematics = np.loadtxt('mouse_med.txt')

    #filtered_1 = signal.filtfilt(b_1, a_1, data_1_kinematics)
    #filtered_med = signal.filtfilt(b_1, a_1, medium_kinematics)
    #filtered_fast = signal.filtfilt(b_fast, a_fast, data_fast_kinematics)
    #filtered_slow = signal.filtfilt(b_slow, a_slow, data_slow_kinematics)

    '''
    b_150, a_150 = signal.butter(10, 20, fs=150, btype='lowpass')
    b_200, a_200 = signal.butter(10, 20, fs=200, btype='lowpass')
    b_250, a_250 = signal.butter(10, 20, fs=250, btype='lowpass')

    mouse_x_sim_150 = np.loadtxt('mouse_150_x.txt')
    filtered_x_sim_150 = signal.filtfilt(b_150, a_150, mouse_x_sim_150)
    mouse_y_sim_150 = np.loadtxt('mouse_150_y.txt')
    filtered_y_sim_150 = signal.filtfilt(b_150, a_150, mouse_y_sim_150)

    mouse_x_sim_200 = np.loadtxt('mouse_200_x.txt')
    filtered_x_sim_200 = signal.filtfilt(b_200, a_200, mouse_x_sim_200)
    mouse_y_sim_200 = np.loadtxt('mouse_200_y.txt')
    filtered_y_sim_200 = signal.filtfilt(b_200, a_200, mouse_y_sim_200)

    mouse_x_sim_250 = np.loadtxt('mouse_250_x.txt')
    filtered_x_sim_250 = signal.filtfilt(b_250, a_250, mouse_x_sim_250)
    mouse_y_sim_250 = np.loadtxt('mouse_250_y.txt')
    filtered_y_sim_250 = signal.filtfilt(b_250, a_250, mouse_y_sim_250)
    '''

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

    scale_sim = 250
    x_sim_offset = 20
    y_sim_offset = 6


    x_sim_150 = (np.sin(np.linspace(-np.pi/2, 3*np.pi/2, 150)) + x_sim_offset) / scale_sim
    y_sim_150 = (np.cos(np.linspace(-np.pi/2, 3*np.pi/2, 150)) + y_sim_offset) / scale_sim

    x_sim_200 = (np.sin(np.linspace(-np.pi/2, 3*np.pi/2, 200)) + x_sim_offset) / scale_sim
    y_sim_200 = (np.cos(np.linspace(-np.pi/2, 3*np.pi/2, 200)) + y_sim_offset) / scale_sim

    x_sim_250 = (np.sin(np.linspace(-np.pi/2, 3*np.pi/2, 250)) + x_sim_offset) / scale_sim
    y_sim_250 = (np.cos(np.linspace(-np.pi/2, 3*np.pi/2, 250)) + y_sim_offset) / scale_sim

    data_fast = np.array(data_fast) / scale - offset
    data_slow = np.array(data_slow) / scale - offset
    data_1 = np.array(data_1) / scale - offset


    if args.plot == 'kinematics':

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
        plt.savefig('scripts/data_all_kinematics_plot.png')
        plt.show()

    elif args.plot == 'x_sim':

        plt.plot(x_sim_150 + .02, label='Simulated', linewidth=4, color='orange')
        plt.plot(filtered_x_sim_150 + .015, label='Model Output', linewidth=4, color='black', linestyle='dashed')

        plt.plot(x_sim_200, linewidth=4, color='orange')
        plt.plot(filtered_x_sim_200, linewidth=4, color='black', linestyle='dashed')

        plt.plot(x_sim_250 - .02, linewidth=4, color='orange')
        plt.plot(filtered_x_sim_250 - .015, linewidth=4, color='black', linestyle='dashed')

        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('X-Position')
        plt.yticks([])
        plt.savefig('data_x_simulated_plot.png')

    elif args.plot == 'y_sim':

        plt.plot(y_sim_150 +.015, label='Simulated', linewidth=4, color='orange')
        plt.plot(filtered_y_sim_150 +.015, label='Model Output', linewidth=4, color='black', linestyle='dashed')

        plt.plot(y_sim_200, linewidth=4, color='orange')
        plt.plot(filtered_y_sim_200, linewidth=4, color='black', linestyle='dashed')

        plt.plot(y_sim_250 -.015, linewidth=4, color='orange')
        plt.plot(filtered_y_sim_250 - .015, linewidth=4, color='black', linestyle='dashed')

        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Y-Position')
        plt.yticks([])
        plt.savefig('data_y_simulated_plot.png')
    
    elif args.plot == 'med':

        plt.plot(data_1, label='Experimental', linewidth=4, color='orange')
        plt.plot(filtered_med, label='Model Output', linewidth=4, color='black', linestyle='dashed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.yticks([])
        plt.savefig('data_med_simulated_plot.png')

if __name__ == '__main__':
    main()