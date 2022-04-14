import pybullet as p
import time
import numpy as np
from numpy.random import rand
import time

import model_utils
from Mouse_RL_Environment import Mouse_Env

file_path = "/files/mouse_test.sdf"
pose_file = "files/locomotion_pose.yaml"

model_offset = (0.0, 0.0, 1.2) #z position modified with global scaling

ctrl = [104, 105, 106, 107, 108, 110, 111]
frame_skip = 1
n_frames = 1
mouseEnv = Mouse_Env(file_path, frame_skip)
model_utils.disable_control(mouseEnv.model)


#Test environment here for now#
p.setTimeStep(.001)
for i in range (1000):
    #posObj = p.getJointState(mouseEnv.model, 106, mouseEnv.client) #not working
    #print(posObj)
    #forces = rand(7)
    forces = np.random.uniform(-.005, .005, size = 7) 
    #print("forces", forces)
    #print('reward', mouseEnv.reward())#doesn't work bc getLinkState doesn't work
    
    mouseEnv.do_simulation(ctrl, n_frames, forces)
    
mouseEnv.close() #disconnects server


#TO-DO: figure out thresholds through testing/ getting position