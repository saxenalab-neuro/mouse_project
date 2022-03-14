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


class Mouse_Env(gym.Env):
    pose_file = "/Users/andreachacon/mouse_biomechanics_paper/data/config/default_pose.yaml"
    def resetPosition(self, pose_file): 
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

    def reset(self):
        self.istep= 0
        self.theta= np.pi
        self.threshold= self.threshold_user
        self.resetPosition()
        return self

    #def reward():
    #    x
        # get xyz of hand, NOT JOINT
        # compare with target XYZ
        # check threshold
        # ex reward: r_x= 1/(1000**d_x)
        # get cost if distance is greater than threshold 



    #def step():
    #    x
        #get reward
        #update target position
        #update joints
        #check steps, etc.
    
    