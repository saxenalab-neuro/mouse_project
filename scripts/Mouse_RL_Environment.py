import datetime
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import itertools
import model_utils as model_utils
import pybullet as p
import pybullet_data
import yaml
import scipy.io

import farms_pylog as pylog
try:
    from farms_muscle.musculo_skeletal_system import MusculoSkeletalSystem
except ImportError:
    pylog.warning("farms-muscle not installed!")
from farms_container import Container

sphere_file = "../files/sphere_small.urdf"

class PyBulletEnv(gym.Env):
    def __init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset):
        #####BUILDS SERVER AND LOADS MODEL#####
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81) #normal gravity
        self.plane = p.loadURDF("plane.urdf") #sets floor
        self.model = p.loadSDF(model_path)[0]#resizes, loads model, returns model id
        self.model_offset = model_offset
        p.resetBasePositionAndOrientation(self.model, self.model_offset, p.getQuaternionFromEuler([0, 0, 80.2])) #resets model position
        self.use_sphere = False
        self.scale = 22
        self.offset = -.6815
        self.muscle_config_file = muscle_config_file
        self.joint_id = {}
        self.link_id = {}
        self.joint_type = {}

        if self.use_sphere:
            self.sphere = p.loadURDF("sphere_small.urdf", globalScaling=.1) #visualizes target position

        self.ctrl = ctrl #control, list of all joints in right arm (shoulder, elbow, wrist + metacarpus for measuring hand pos)
        self.pose_file = pose_file
        
        #####MUSCLES + DATA LOGGING#####
        self.container = Container(max_iterations=int(1000000))

        # Physics simulation to namespace
        self.sim_data = self.container.add_namespace('physics')

        self.initialize_muscles()
        model_utils.reset_model_position(self.model, self.pose_file)
        self.container.initialize()
        #self.muscles.setup_integrator()

        #####META PARAMETERS FOR SIMULATION#####
        self.n_fixedsteps= 20
        self._max_episode_steps = timestep #Does not matter. It is being set in the main.py where the total number of steps are being changed.
        self.threshold_user = 0.0045
        self.timestep = timestep
        self.frame_skip= frame_skip

        #####TARGET POSITION USING POINT IN SPACE: X, Y, Z#####
        ###x, y, z for initializing from hand starting position, target_pos for updating
        self.x_pos = [0]
        self.y_pos = p.getLinkState(self.model, 115)[0][1]
        self.z_theta = np.linspace(0, 2*np.pi, (self.timestep - 20) // 3)
        self.starting_z = [0] * 20
        self.z_theta_cycle = [*self.starting_z, *self.z_theta, *self.z_theta, *self.z_theta]
        self.z_pos = (np.sin(self.z_theta[0]) + 11) / 500

        self.target_pos = [self.x_pos[0]/self.scale - self.offset, self.y_pos, self.z_pos]


        if self.use_sphere:
            p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))

        p.resetDebugVisualizerCamera(0.3, 15, -10, [0, 0.21, 0])

        self.action_space = spaces.Box(low=np.ones(18), high=np.ones(18), dtype=np.float32)

        self.seed()

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_ids(self):
        return self.client, self.model
    
    #####OVERWRITTEN IN CHILD CLASS#####
    def reset_model(self, pose_file):
        raise NotImplementedError

    def reset(self, pose_file):
        self.istep = 0
        model_utils.disable_control(self.model) #disables torque/position
        self.reset_model(pose_file) #resets model position
        self.container.initialize() #resets container
        self.muscles.setup_integrator() #resets muscles
        #resets target position
        self.z_theta = np.linspace(0, 2*np.pi, (self.timestep - 20) // 3)
        self.starting_z = [0] * 20
        self.z_theta_cycle = [*self.starting_z, *self.z_theta, *self.z_theta, *self.z_theta]
        self.z_pos = (np.sin(self.z_theta[0]) + 11) / 500
        self.target_pos = [self.x_pos[0]/self.scale -self.offset, self.y_pos, self.z_pos]
        if self.use_sphere:
            p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))
        
        self.threshold_x = self.threshold_user #resets threshold
        self.threshold_y = self.threshold_user
        self.threshold_z = self.threshold_user
    
    def initialize_muscles(self):
        self.muscles = MusculoSkeletalSystem(self.container, 1e-3, self.muscle_config_file)

    def do_simulation(self):
        self.muscles.step()
        self.container.update_log()
        p.stepSimulation()
    
    #####DISCONNECTS SERVER#####
    def close(self):
        p.disconnect(self.client)

class Mouse_Env(PyBulletEnv):

    def __init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset):
        PyBulletEnv.__init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset)
        u = self.container.muscles.activations
        self.muscle_params = {}
        self.muscle_excitation = {}
        for muscle in self.muscles.muscles.keys():
               self.muscle_params[muscle] = u.get_parameter('stim_{}'.format(muscle))
               self.muscle_excitation[muscle] = p.addUserDebugParameter("flexor {}".format(muscle), 0, 1, 0.00)
               self.muscle_params[muscle].value = 0

    def reset_model(self, pose_file): 
        model_utils.reset_model_position(self.model, pose_file)

    def get_cost(self, forces):
        scaler= 1/50
        cost = scaler * np.sum(np.abs(forces))
        return cost

    def get_reward(self): 
        hand_pos = p.getLinkState(self.model, 115)[0] #(x, y, z)

        d_x = np.abs(hand_pos[0] - self.target_pos[0])
        d_y = np.abs(hand_pos[1] - self.target_pos[1])
        d_z = np.abs(hand_pos[2] - self.target_pos[2])

        distances = [d_x, d_y, d_z]

        if d_x > self.threshold_x or d_y > self.threshold_y or d_z > self.threshold_z:
            reward = -5
        
        else:
            r_x= 1/(2500**d_x)
            r_y= 1/(2500**d_y)
            r_z= 1/(2500**d_z)

            reward= r_x + r_y + r_z

        return reward, distances

    def is_done(self):
        hand_pos =  np.array(p.getLinkState(self.model, 115)[0]) #(x, y, z)
        criteria = hand_pos - self.target_pos

        if self.istep < self.timestep:
            if np.abs(criteria[0]) > self.threshold_x or np.abs(criteria[1]) > self.threshold_y or np.abs(criteria[2]) > self.threshold_z:
                return True
            else:
                return False
        else:
            return True

    def update_target_pos(self):
        self.target_pos = [self.x_pos[(self.istep-1)]/self.scale-self.offset, self.y_pos, ((np.sin(self.z_theta_cycle[self.istep-1])+11)/500)]

        if self.use_sphere:
            p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))
        
    def get_joint_positions_and_velocities(self):
        joint_positions = []
        joint_velocities = []
        for i in range(len(self.ctrl)):
            joint_positions.append(p.getJointState(self.model, self.ctrl[i])[0])
            joint_velocities.append(p.getJointState(self.model, self.ctrl[i])[1]/100)
        return joint_positions, joint_velocities

    def update_state(self, act, joint_positions, joint_velocities, target_velocity, distances):
        state = [*list(act), *list(joint_positions), *list(joint_velocities), *list(self.target_pos), *list(target_velocity), *list(distances)]
        return state

    def get_cur_state(self):

        joint_positions, _ = self.get_joint_positions_and_velocities()
        _, distance = self.get_reward()
        return [*list(self.get_activations()), *list(joint_positions), *[0., 0., 0., 0., 0., 0., 0.], *list(self.target_pos), *[0, 0, 0], *distance]
    
    def controller_to_actuator(self, forces):

        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_AN", forces[0])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_BBL",forces[1])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_BBS", forces[2])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_BRA", forces[3])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_COR", forces[4])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_ECRB", forces[5])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_ECRL", forces[6])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_ECU", forces[7])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_EIP1", forces[8])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_EIP2", forces[9])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_FCR", forces[10])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_FCU", forces[11])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_PLO", forces[12])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_PQU", forces[13])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_PTE", forces[14])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_TBL", forces[15])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_TBM", forces[16])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_TBO", forces[17])

    def get_activations(self):
        activations = []

        activations.append(self.container.muscles.states.get_parameter_value('activation_RIGHT_FORE_AN'))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_BBL"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_BBS"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_BRA"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_COR"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_ECRB"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_ECRL"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_ECU"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_EIP1"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_EIP2"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_FCR"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_FCU"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_PLO"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_PQU"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_PTE"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_TBL"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_TBM"))
        activations.append(self.container.muscles.states.get_parameter_value("activation_RIGHT_FORE_TBO"))

        return activations

    def step(self, forces, i_episode):
        self.istep += 1

        self.controller_to_actuator(forces)

        #can edit threshold with episodes
        if self.istep > self.n_fixedsteps:
            if i_episode < self.timestep // 3:
                self.threshold_x = .006
                self.threshold_y = .006
                self.threshold_z = .006
            elif i_episode > self.timestep // 3:
                self.threshold_x = .004
                self.threshold_y = .004
                self.threshold_z = .004

        self.do_simulation()

        act = self.get_activations()
        reward, distances = self.get_reward()
        cost = self.get_cost(forces)
        final_reward= (5*reward) - (cost)

        done = self.is_done()
        
        prev_target = np.array(self.target_pos)
        self.update_target_pos()
        curr_target = np.array(self.target_pos)

        target_vel = (curr_target - prev_target) / (.001) #need clarification about dt
        joint_positions, joint_velocities = self.get_joint_positions_and_velocities()

        state = self.update_state(act, joint_positions, joint_velocities, target_vel, distances)

        return state, final_reward, done
