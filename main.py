import pybullet as p
import numpy as np
import time

import model_utils
from Mouse_RL_Environment import Mouse_Env
from Mouse_Stabilize_Environment import Mouse_Stability_Env

file_path = "/Users/andreachacon/Documents/GitHub/mouse_project/files/mouse_test.sdf" ###All joints fixed except right arm, left wrist, & knees
#file_path = "/Users/andreachacon/Documents/GitHub/mouse_project/files/mouse_test.sdf" ###All joints fixed except right arm
pose_file = "/Users/andreachacon/Documents/GitHub/mouse_project/files/default_pose.yaml"
muscle_config_file = "/Users/andreachacon/Documents/GitHub/mouse_project/files/right_forelimb.yaml"

model_offset = (0.0, 0.0, 1.2) #z position modified with global scaling

#ARM CONTROL
ctrl = [104, 105, 106, 107, 108, 110, 111]

#STABILITY CONTROL
#ctrl = [142, 125, 91, 92, 104, 105, 106, 107, 108, 110, 111]

###JOINT TO INDEX###
#RKnee - 142
#LKnee - 125
#LWrist_adduction - 91
#LWrist_flexion - 92
#RShoulder_rotation - 104
#RShoulder_adduction - 105
#RShoulder_flexion - 106
#RElbow_flexion - 107
#RElbow_supination - 108
#RWrist_adduction - 110
#RWrist_flexion - 111
#Lumbar2_bending - 12, use link(lumbar 1) for stability reward


###PARAMETERS###
frame_skip = 1
n_frames = 1
timestep = 1000

#ARM ENV
mouseEnv = Mouse_Env(file_path, muscle_config_file, frame_skip, ctrl, timestep)

#STABILITY ENV
#mouseEnv = Mouse_Stability_Env(file_path, muscle_config_file, frame_skip, ctrl, timestep)

model_utils.disable_control(mouseEnv.model)

p.setTimeStep(.001)
mouseEnv.reset(pose_file)

for i in range (mouseEnv.timestep):
    #ARM TRAINING
    forces = np.random.uniform(-.005, .005, size = 7) 

    #STABILITY TRAINING
    #forces = np.random.uniform(-.005, .005, size = 4) #random activations to knees, LWrist
    #forces.append([0, 0, 0, 0, 0, 0, 0]) #no activations to right arm


    state, final_reward, done = mouseEnv.step(forces)
    #print("reward", final_reward, "| is not done?", done)  11
    #print("hand pos", p.getLinkState(modelid, 112)[0])
    #print(state)
    
mouseEnv.close() #disconnects server

