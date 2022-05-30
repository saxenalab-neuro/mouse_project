import pybullet as p
import time
import numpy as np
import time

import model_utils
from Mouse_RL_Environment import Mouse_Env

file_path = "/Users/andreachacon/Documents/GitHub/mouse_project/files/mouse_test.sdf" ###Changed joints to fixed: model starts floating due to 0-gravity
#file_path = "/files/mouse_with_joint_limits.sdf" ###Unfixed joints, no issues with 0-gravity
pose_file = "/Users/andreachacon/Documents/GitHub/mouse_project/files/default_pose.yaml"
muscle_config_file = "/Users/andreachacon/Documents/GitHub/mouse_project/files/right_forelimb.yaml"

model_offset = (0.0, 0.0, 1.2) #z position modified with global scaling

ctrl = [104, 105, 106, 107, 108, 110, 111] #7
frame_skip = 1
n_frames = 1
timestep = 1000
mouseEnv = Mouse_Env(file_path, muscle_config_file, frame_skip, ctrl, timestep)
model_utils.disable_control(mouseEnv.model)

#FINDING HAND STARTING POS
#clientid, modelid = mouseEnv.get_ids()
#print("hand starting pos", p.getLinkState(modelid, 112)[0])
#HAND STARTING POS: (1.3697159804379864, -0.09075569325649711, 0.2675971224717795)


p.setTimeStep(.001)
mouseEnv.reset(pose_file)

for i in range (mouseEnv.timestep):
    forces = np.random.uniform(-.005, .005, size = 7)
    final_reward, done = mouseEnv.step(forces)
    #print("reward", final_reward, "| is not done?", done)
    #print("hand pos", p.getLinkState(modelid, 112)[0])
    
mouseEnv.close() #disconnects server

