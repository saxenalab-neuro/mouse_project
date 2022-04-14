import datetime
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import itertools
import torch
import pybullet as p
import pybullet_data
import yaml
from model_utils import cart2sph
from model_utils import sph2cart

file_path = "/files/mouse_with_joint_limits.sdf"
pose_file = "files/locomotion_pose.yaml"

model_offset = (0.0, 0.0, 1.2) #z position modified with global scaling

ctrl = [104, 105, 106, 107, 108, 110, 111]
#RShoulder_rotation - 104
#RShoulder_adduction - 105
#RShoulder_flexion - 106
#RElbow_flexion - 107
#RElbow_supination - 108
#RWrist_adduction - 110
#RWrist_flexion - 111
#RMetacarpus1_flextion - 112, use link (carpus) for pos

class PyBulletEnv(gym.Env):
    def __init__(self, model_path, frame_skip, ctrl):
        #####BUILDS SERVER AND LOADS MODEL#####
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,0) #no gravity
        self.plane = p.loadURDF("plane.urdf")
        self.model = p.loadSDF(model_path, globalScaling = 25)[0]# resizes
        self.frame_skip= frame_skip
        p.resetBasePositionAndOrientation(self.model, model_offset, p.getQuaternionFromEuler([0, 0, 80.2]))

        self.ctrl = ctrl
        
        #####TARGET POSITION USING POINT IN SPACE: theta, rho, phi#####
        ###theta,rho,phi for initializing
        #target_pos for updating
        self.theta = np.pi #FIGURE OUT STARTING POSITION
        self.rho = 0 #FIGURE OUT STARTING POSITION
        self.phi = 0 #FIGURE OUT STARTING POSITION
        self.target_pos = [self.theta, self.rho, self.phi]

        #####META PARAMETERS FOR SIMULATION#####
        self.n_fixedsteps= 20
        self.timestep_limit= (1319 * 1) + self.n_fixedsteps
        # self._max_episode_steps= self.timestep_limit/ 2
        self._max_episode_steps= 1000   #Do not matter. It is being set in the main.py where the total number of steps are being changed.
        self.threshold = .03 #CAN BE EDITED


        #initialize neural networks here

        #self.seed()

    #def seed(self, seed=None):
     #   self.np_random, seed = seeding.np_random(seed)
      #  return [seed]

    def get_ids(self):
        return self.client, self.model
    
    #####OVERWRITTEN IN CHILD CLASS#####
    def reset_model(self):
        raise NotImplementedError

    def reset(self):
        self.istep= 0
        self.theta= np.pi#FIND STARTING POS
        self.rho = 0 #FIND STARTING POS
        self.phi = 0 #FIND STARTING POS
        self.target_pos = [self.theta, self.rho, self.phi]
        self.threshold= self.threshold_user
        self.reset_model()

    def do_simulation(self, n_frames, forcesArray):
        for _ in range(n_frames):
            p.setJointMotorControlArray(self.model, self.ctrl, p.TORQUE_CONTROL, forces = forcesArray)
            p.stepSimulation()

    #Might need to be adjusted
    def get_cost(self, forces):
        scaler= 1/50
        act = forces
        cost = scaler * np.sum(np.abs(act))
        return cost

    #####DISCONNECTS SERVER#####
    def close(self):
        p.disconnect(self.client)

class Mouse_Env(PyBulletEnv):

    def __init__(self, model_path, frame_skip):
        PyBulletEnv.__init__(self, model_path, frame_skip)

    def reset_model(self, pose_file): 
        joint_list = []
        for joint in range(p.getNumJoints(self.model)):
            joint_list.append(joint)
        with open(pose_file) as stream:
            data = yaml.load(stream, Loader=yaml.SafeLoader)
            data = {k.lower(): v for k, v in data.items()}
        for joint in joint_list:
            _pose = np.deg2rad(data.get(p.getJointInfo(self, joint)[1].decode('UTF-8').lower(), 0))#decode removes b' prefix
            p.resetJointState(self, joint, targetValue=_pose)

    def reward(self): 
        hand_pos = p.getLinkState(self.model, 112)[0] #(x, y, z)
        theta, rho, phi= sph2cart(self.target_pos[0], self.target_pos[1], self.target_pos[2])

        d_x = np.abs(hand_pos[0] - theta)
        d_y = np.abs(hand_pos[1] - rho)
        d_z = np.abs(hand_pos[2] - phi)

        if d_x > self.threshold or d_y > self.threshold or d_z > self.threshold:
            return -5

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
        #CURRENTLY HAS ARBITRARY VALUES
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

    def step(self, forces):
        self.istep += 1

        #can edit threshold with episodes
        if self.istep > self.n_fixedsteps:
            self.threshold = 0.032

        self.do_simulation(self.frame_skip, forces)

        reward= self.reward()

        #can play around with cost/reward value
        cost= self.get_cost(forces)
        final_reward= (5*reward) - (0.5*cost)

        self.update_target_pos()

        done= self.is_done()

        return final_reward, done
    
#ISSUES:
# getLinkState/getJointState failing
# seeding fails 

#TO_DO:
# probably need to add camera at some point
# add initialization of neural networks when written (add states)
# play with threshold
# add maximums for circular motion(parameters)
# write render, set state, dt
# learn parameters so rest of mouse doesn't move
# learn position/pose to reset/intialize to
# resource: https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e
# wrapper: https://blog.paperspace.com/getting-started-with-openai-gym/