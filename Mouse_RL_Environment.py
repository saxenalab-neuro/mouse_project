import datetime
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import itertools
import torch
import model_utils
import pybullet as p
import pybullet_data
import yaml

import farms_pylog as pylog
try:
    from farms_muscle.musculo_skeletal_system import MusculoSkeletalSystem
except ImportError:
    pylog.warning("farms-muscle not installed!")
from farms_container import Container

sphere_file = "/Users/andreachacon/Documents/GitHub/mouse_project/files/sphere_small.urdf"

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
    def __init__(self, model_path, muscle_config_file, frame_skip, ctrl, timestep):
        #####BUILDS SERVER AND LOADS MODEL#####
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8) #normal gravity
        self.plane = p.loadURDF("plane.urdf") #sets floor
        self.model = p.loadSDF(model_path, globalScaling = 25)[0]#resizes, loads model, returns model id
        p.resetBasePositionAndOrientation(self.model, model_offset, p.getQuaternionFromEuler([0, 0, 80.2])) #resets model position
        self.sphere = p.loadURDF("sphere_small.urdf", globalScaling = 2)
        group = 0#other objects don't collide with me
        mask=0 # don't collide with any other object
        p. setCollisionFilterGroupMask(self.sphere, 0, group, mask)
        

        self.ctrl = ctrl #control, list of all joints in right arm (shoulder, elbow, wrist + metacarpus for measuring hand pos)
        
        #####MUSCLES#####
        #self.container = Container(max_iterations=int(2.5/0.001))
        #self.container.initialize()
        #self.muscles = MusculoSkeletalSystem(self.container, 1e-3, muscle_config_file)
        #self.muscles.print_system() 
        #print("num states", self.muscles.muscle_sys.num_states)
        #self.muscles.setup_integrator()

        #####META PARAMETERS FOR SIMULATION#####
        self.n_fixedsteps= 20
        self.timestep_limit= (1319 * 1) + self.n_fixedsteps
        # self._max_episode_steps= self.timestep_limit/ 2
        self._max_episode_steps= 1000 #Does not matter. It is being set in the main.py where the total number of steps are being changed.
        self.threshold_user= 0.064
        self.timestep = timestep
        self.frame_skip= frame_skip

        #####TARGET POSITION USING POINT IN SPACE: X, Y, Z#####
        ###x, y, z for initializing from hand starting position, target_pos for updating
        self.x_pos = p.getLinkState(self.model, 112)[0][0]
        self.y_pos = p.getLinkState(self.model, 112)[0][1]
        self.z_pos = p.getLinkState(self.model, 112)[0][2]
        self.target_pos = [self.x_pos, self.y_pos, self.z_pos]
        self.center = [self.x_pos, self.y_pos, self.z_pos]
        self.radius = .5 #arbitrarily defined
        self.theta = np.linspace(0, 2 * np.pi, self.timestep) #array from 0-2pi of timestep values
        
        p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))

        #self.seed()

    #def seed(self, seed = None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]

    def get_ids(self):
        return self.client, self.model
    
    #####OVERWRITTEN IN CHILD CLASS#####
    def reset_model(self, pose_file):
        raise NotImplementedError

    def reset(self, pose_file):
        self.istep= 0
        #carpus starting position, from getLinkState of metacarpus1
        self.target_pos = p.getLinkState(self.model, 112)[0]
        self.threshold= self.threshold_user 
        self.reset_model(pose_file)

    def do_simulation(self, n_frames, forcesArray):
        for _ in range(n_frames):
            p.setJointMotorControlArray(self.model, self.ctrl, p.TORQUE_CONTROL, forces = forcesArray)
            p.stepSimulation()

    #####DISCONNECTS SERVER#####
    def close(self):
        p.disconnect(self.client)

class Mouse_Env(PyBulletEnv):

    def __init__(self, model_path, muscle_config_file, frame_skip, ctrl, timestep):
        PyBulletEnv.__init__(self, model_path, muscle_config_file, frame_skip, ctrl, timestep)
        #u = self.container.muscles.activations
        #for muscle in self.muscles.muscles.keys():
        #       self.muscle_params[muscle] = u.get_parameter('stim_{}'.format(muscle))
        #       self.muscle_excitation[muscle] = p.addUserDebugParameter("flexor {}".format(muscle), 0, 1, 0.00)
        #       self.muscle_params[muscle].value = 0

    def reset_model(self, pose_file): 
        model_utils.reset_model_position(self.model, pose_file)

    def get_cost(self, forces):
        scaler= 1/50
        cost = scaler * np.sum(np.abs(forces))
        return cost

    def get_reward(self): 
        hand_pos = p.getLinkState(self.model, 112)[0] #(x, y, z)

        d_x = np.abs(hand_pos[0] - self.target_pos[0])
        d_y = np.abs(hand_pos[1] - self.target_pos[1])
        d_z = np.abs(hand_pos[2] - self.target_pos[2])

        if d_x > self.threshold or d_y > self.threshold or d_z > self.threshold:
            return -5

        r_x= 1/(1000**d_x)
        r_y= 1/(1000**d_y)
        r_z= 1/(1000**d_z)

        reward= r_x + r_y + r_z

        return reward

    def is_done(self):
        hand_pos =  np.array(p.getLinkState(self.model, 112)[0]) #(x, y, z)
        criteria = hand_pos - self.target_pos

        if self.istep < self.timestep_limit:
            if np.abs(criteria[0]) > self.threshold or np.abs(criteria[1]) > self.threshold or np.abs(criteria[2]) > self.threshold:
                return True
            else:
                return False
        else:
            return True

    def update_target_pos(self):

        self.x_pos = self.radius * np.cos(self.theta[self.istep - 1]) + self.center[0]
        self.z_pos = self.radius * np.sin(self.theta[self.istep - 1]) + self.center[2]
        self.target_pos = [self.x_pos, self.y_pos, self.z_pos]
        p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))
        #print("x, y, z", self.target_pos)


    def step(self, forces):
        self.istep += 1

        #can edit threshold with episodes
        if self.istep > self.n_fixedsteps:
            self.threshold = 0.032

        self.do_simulation(self.frame_skip, forces)
        #self.muscles.step()

        reward= self.get_reward()
        cost= self.get_cost(forces)
        final_reward= (5*reward) - (0.5*cost)

        done= self.is_done()
        
        self.update_target_pos()

        
        return final_reward, done
#to_do:
# things to hold model down