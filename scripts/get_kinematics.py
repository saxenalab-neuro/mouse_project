import numpy as np
import matplotlib.pyplot as plt
import scipy

data_1_kinematics = np.loadtxt('mouse_1.txt')
data_fast_kinematics = np.loadtxt('mouse_fast.txt')
data_slow_kinematics = np.loadtxt('mouse_slow.txt')

##################### DATA FAST ############################
mat = scipy.io.loadmat('../data/kinematics_session_mean_alt_fast.mat')
data = np.array(mat['kinematics_session_mean'][2])
data_fast = data[231:401:1] * -1
data_fast = [-13.45250312, *data_fast[8:]]

#################### DATA SLOW #########################
mat = scipy.io.loadmat('../data/kinematics_session_mean_alt_slow.mat')
data = np.array(mat['kinematics_session_mean'][2])
data_slow = data[256:476:1] * -1

#################### DATA_1 #########################
mat = scipy.io.loadmat('../data/kinematics_session_mean_alt1.mat')
data = np.array(mat['kinematics_session_mean'][2])
data_1= data[226:406:1] * -1
data_1 = [-13.45250312, *data_1[4:]]

scale = 21
offset = -.71

data_fast = np.array(data_fast) / scale - offset
data_slow = np.array(data_slow) / scale - offset
data_1 = np.array(data_1) / scale - offset

plt.plot(data_1, label='Data 1')
plt.plot(data_1_kinematics, label='Data 1 Mouse')
plt.legend()
plt.savefig('data_1_kinematics_plot.png')
