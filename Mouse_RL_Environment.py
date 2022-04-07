import datetime
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import itertools
import torch
import pybullet as p
import pybullet_envs
import yaml

arm_indexes = [104, 105, 106, 107, 108, 110, 111, 112]
#RShoulder_rotation - 104
#RShoulder_adduction - 105
#RShoulder_flexion - 106
#RElbow_flexion - 107
#RElbow_supination - 108
#RWrist_adduction - 110
#RWrist_flexion - 111
#RMetacarpus1_flextion - 112, use link (carpus) for pos

#helper functions, possibly to be added to a util file
def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z) #rho
    el = np.arctan2(z, hxy) #theta
    az = np.arctan2(y, x) #phi
    return az, el, r

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

class PyBulletEnv(gym.env):
    def __init__(self, modelId, frame_skip):
        self.frame_skip= frame_skip
        self.model = modelId

        #using point in space: theta, rho, phi
        theta = np.pi #FIGURE OUT STARTING POSITION
        rho = 0 #FIGURE OUT STARTING POSITION
        phi = 0 #FIGURE OUT STARTING POSITION
        self.target_pos = [theta, rho, phi]

        #Meta parameters for the simulation
        self.n_fixedsteps= 20
        self.timestep_limit= (1319 * 1) + self.n_fixedsteps
        # self._max_episode_steps= self.timestep_limit/ 2
        self._max_episode_steps= 1000   #Do not matter. It is being set in the main.py where the total number of steps are being changed.
        self.threshold = .03

    def reset_model(self):
        raise NotImplementedError

    def reset(self):
        self.istep= 0
        self.theta= np.pi
        self.threshold= self.threshold_user
        self.resetPosition()
        return self


class Mouse_Env(PyBulletEnv):

    def __init__(self, mouseId, frame_skip):
        PyBulletEnv.__init__(self, mouseId, frame_skip)

    pose_file = "/files/default_pose.yaml"
    def reset_model(self, pose_file): 
        joint_list = []
        for joint in range(p.getNumJoints(self, )):
            joint_list.append(joint)
        with open(pose_file) as stream:
            data = yaml.load(stream, Loader=yaml.SafeLoader)
            data = {k.lower(): v for k, v in data.items()}
            #print(data)
        for joint in joint_list:
            #print(data.get(p.getJointInfo(boxId, joint)[1]))
            joint_name =p.getJointInfo(self, joint)[1] 
            _pose = np.deg2rad(data.get(p.getJointInfo(self, joint)[1].decode('UTF-8').lower(), 0))#decode removes b' prefix
            #print(p.getJointInfo(boxId, joint)[1].decode('UTF-8').lower(), _pose)
            p.resetJointState(self, joint, targetValue=_pose)

    def reward(self): 
        hand_pos = p.getLinkState(self.model, 112)[0] #(x, y, z)
        x, y, z= sph2cart(self.target_pos[0], self.target_pos[1], self.target_pos[2])

        d_x = np.abs(hand_pos[0] - x)
        d_y = np.abs(hand_pos[1] - y)
        d_z = np.abs(hand_pos[2] - z)

        if d_x > self.threshold or d_y > self.threshold or d_z > self.threshold:
            return -5


        #can play around with these functions later
        r_x= 1/(1000**d_x)
        r_y= 1/(1000**d_y)
        r_z= 1/(1000**d_z)

        reward= r_x + r_y + r_z

        return reward

    def is_done(self):
        x, y, z= sph2cart(self.target_pos[0], self.target_pos[1], self.target_pos[2])
        target = [x, y, z]
        hand_pos =hand_pos = p.getLinkState(self.model, 112)[0] #(x, y, z)
        criteria= hand_pos - target

        if self.istep < self.timestep_limit:
            if np.abs(criteria[0]) > self.threshold or np.abs(criteria[1]) > self.threshold or np.abs(criteria[2]) > self.threshold:
                return True
            else:
                return False
        else:
            return True

    def update_target_pos(self):
        #depends on how fast we want it to move, play around with values
        if(self.target_pos[0] < self.max_theta):
            self.target_pos[0] += np.pi/6
        else:
            self.target_pos[0] -= np.pi/6

        if(self.target_pos[1] < self.max_rho):
                self.target_pos[1] += np.pi/6
        else:
            self.target_pos[1] -= np.pi/6

        if(self.target_pos[2] < self.max_phi):
                self.target_pos[0] += np.pi/6
        else:
            self.target_pos[2] -= np.pi/6
    
    def update_joints(self):
        for x in arm_indexes:
            p.setJointMotorControl(self.model, x, p.TORQUE_CONTROL, force = .002)
            #THIS IS TEMPORARY, NEED TO FIGURE OUT WHAT WORKS

    def step(self, action):
        self.istep += 1

        #can edit threshold with episodes
        if self.istep > self.n_fixedsteps:
            self.threshold = 0.032

        self.do_simulation(action, self.frame_skip)

        reward= self.reward()

        #can play around with cost/reward value
        cost= self.get_cost(action)
        final_reward= (5*reward) - (0.5*cost)

        self.update_target_pos()
        self.update_joints()

        done= self.is_done()
        

        return final_reward, done
        #get reward
        #need to figure out how to edit torque here
    

#TO_DO:
# probably need to add camera at some point
# add initialization of neural networks when written
# play with threshold
# add maximums for circular motion
# write do_simulation-- how to move it?
# need action space and observation space