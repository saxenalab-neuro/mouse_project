import datetime
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import itertools
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

sphere_file = "../files/sphere_small.urdf"

class PyBulletEnv(gym.Env):
    def __init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset):
        #####BUILDS SERVER AND LOADS MODEL#####
        #self.client = p.connect(p.GUI)
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8) #normal gravity
        self.plane = p.loadURDF("plane.urdf") #sets floor
        self.model = p.loadSDF(model_path)[0]#resizes, loads model, returns model id
        self.model_offset = model_offset
        p.resetBasePositionAndOrientation(self.model, self.model_offset, p.getQuaternionFromEuler([0, 0, 80.2])) #resets model position
        self.stability = p.getLinkState(self.model, 12)[0]
        self.use_sphere = True

        if self.use_sphere:
            self.sphere = p.loadURDF("sphere_small.urdf", globalScaling=.1) #visualizes target position

        self.ctrl = ctrl #control, list of all joints in right arm (shoulder, elbow, wrist + metacarpus for measuring hand pos)
        self.pose_file = pose_file
        
        #####MUSCLES + DATA LOGGING#####
        self.container = Container(max_iterations=int(2.5/0.001))

        # Physics simulation to namespace
        self.sim_data = self.container.add_namespace('physics')
        self.create_tables()

        self.muscles = MusculoSkeletalSystem(self.container, 1e-3, config_path=muscle_config_file)

        model_utils.reset_model_position(self.model, self.pose_file)
        
        self.container.initialize()
        self.muscles.print_system() 
        print("num states", self.muscles.muscle_sys.num_states)
        self.muscles.setup_integrator()

        #####META PARAMETERS FOR SIMULATION#####
        self.n_fixedsteps= 20
        self.timestep_limit = 50
        # self._max_episode_steps= self.timestep_limit/ 2
        self._max_episode_steps = timestep #Does not matter. It is being set in the main.py where the total number of steps are being changed.
        self.threshold_user= 0.008
        self.timestep = timestep
        self.frame_skip= frame_skip

        #####TARGET POSITION USING POINT IN SPACE: X, Y, Z#####
        ###x, y, z for initializing from hand starting position, target_pos for updating
        self.x_pos = p.getLinkState(self.model, 112)[0][0]
        self.orig_x = self.x_pos
        self.y_pos = p.getLinkState(self.model, 112)[0][1]
        self.orig_y = self.y_pos
        self.z_pos = p.getLinkState(self.model, 112)[0][2]
        self.orig_z = self.z_pos

        self.radius = .0045
        self.theta = np.linspace(np.pi, -np.pi, self.timestep) #array from 0-2pi of timestep values
        self.center = [self.x_pos + .003, self.y_pos, self.z_pos + .005]
        self.target_pos = [self.radius * np.cos(self.theta[0]) + self.center[0], self.y_pos, self.radius * np.sin(self.theta[0]) + self.center[2]]
        if self.use_sphere:
            p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))

        p.resetDebugVisualizerCamera(.6, 50, -35, [-.25, 0.21, -0.23])

        self.action_space = spaces.Box(low=np.array([-.05,-.05,-.05,-.05,-.05,-.05,-.05]), high=np.array([.05,.05,.05,.05,.05,.05,.05]), dtype=np.float32)
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
        #carpus starting position, from getLinkState of metacarpus1
        self.reset_model(pose_file)
        self.target_pos = [self.radius * np.cos(self.theta[0]) + self.center[0], self.y_pos, self.radius * np.sin(self.theta[0]) + self.center[2]]
        if self.use_sphere:
            p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))
        self.threshold = self.threshold_user 

    def do_simulation(self, n_frames, forcesArray):
        for _ in range(n_frames):
            p.setJointMotorControlArray(self.model, self.ctrl, p.TORQUE_CONTROL, forces = forcesArray)
            p.resetBasePositionAndOrientation(self.model, self.model_offset, p.getQuaternionFromEuler([0, 0, 80.2]))

    #####DISCONNECTS SERVER#####
    def close(self):
        p.disconnect(self.client)

    def create_tables(self):
        ####ADD TABLES TO CONTAINER###
        self.sim_data.add_table('joint_positions')
        self.sim_data.add_table('joint_velocity')
        self.sim_data.add_table('target_positions')
        self.sim_data.add_table('target_velocity')
        self.sim_data.add_table('distances')

        ###ADD PARAMETERS TO TABLES###
        self.sim_data.joint_positions.add_parameter('RShoulder_rotation')
        self.sim_data.joint_positions.add_parameter('RShoulder_adduction')
        self.sim_data.joint_positions.add_parameter('RShoulder_flexion')
        self.sim_data.joint_positions.add_parameter('RElbow_flexion')
        self.sim_data.joint_positions.add_parameter('RElbow_supination')
        self.sim_data.joint_positions.add_parameter('RWrist_adduction')
        self.sim_data.joint_positions.add_parameter('RWrist_flexion')

        self.sim_data.joint_velocity.add_parameter('RShoulder_rotation')
        self.sim_data.joint_velocity.add_parameter('RShoulder_adduction')
        self.sim_data.joint_velocity.add_parameter('RShoulder_flexion')
        self.sim_data.joint_velocity.add_parameter('RElbow_flexion')
        self.sim_data.joint_velocity.add_parameter('RElbow_supination')
        self.sim_data.joint_velocity.add_parameter('RWrist_adduction')
        self.sim_data.joint_velocity.add_parameter('Rwrist_flexion')

        self.sim_data.target_positions.add_parameter('x')
        self.sim_data.target_positions.add_parameter('y')
        self.sim_data.target_positions.add_parameter('z')

        self.sim_data.target_velocity.add_parameter('Carpus_velocity')

        self.sim_data.distances.add_parameter('x')
        self.sim_data.distances.add_parameter('y')
        self.sim_data.distances.add_parameter('z')
   
    def update_logs(self, joint_positions, joint_velocities, target_velocity, distances):

        self.sim_data.joint_positions.values = np.asarray(joint_positions)
        self.sim_data.joint_velocity.values = np.asarray(joint_velocities)
        self.sim_data.target_positions.values = np.asarray(self.target_pos)
        self.sim_data.target_velocity.values = np.asarray(target_velocity)
        self.sim_data.distances.values = np.asarray(distances)

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
        hand_pos = p.getLinkState(self.model, 112)[0] #(x, y, z)

        d_x = np.abs(hand_pos[0] - self.target_pos[0])
        d_y = np.abs(hand_pos[1] - self.target_pos[1])
        d_z = np.abs(hand_pos[2] - self.target_pos[2])

        distances = [d_x, d_y, d_z]

        if d_x > self.threshold or d_y > self.threshold or d_z > self.threshold:
            reward = -5
        
        else:
            r_x= 1/(1000**d_x)
            r_y= 1/(1000**d_y)
            r_z= 1/(1000**d_z)

            reward= r_x + r_y + r_z

        return reward, distances

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

        if self.use_sphere:
            p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))
        #print("x, y, z", self.target_pos)

    def get_joint_positions_and_velocities(self):
        joint_positions = []
        joint_velocities = []
        for i in range(len(self.ctrl)):
            joint_positions.append(p.getJointState(self.model, self.ctrl[i])[0])
            joint_velocities.append(p.getJointState(self.model, self.ctrl[i])[1])
        return joint_positions, joint_velocities

    def update_state(self, joint_positions, joint_velocities, target_velocity, distances):
        state = [*list(joint_positions), *list(joint_velocities), *list(self.target_pos), *list(target_velocity), *list(distances)]
        return state

    def get_cur_state(self):

        joint_positions, joint_velocities = self.get_joint_positions_and_velocities()
        _, distance = self.get_reward()
        return [*list(joint_positions), *list(joint_velocities), *list(self.target_pos), *[0, 0, 0], *distance]

    def step(self, forces):

        self.istep += 1

        #can edit threshold with episodes
        if self.istep > self.n_fixedsteps:
            self.threshold = 0.008

        self.do_simulation(self.frame_skip, forces)
        self.muscles.step()
        
        reward, distances = self.get_reward()
        cost = self.get_cost(forces)
        final_reward= (5*reward) - (.5*cost)

        done = self.is_done()
        
        prev_target = np.array(self.target_pos)
        self.update_target_pos()
        curr_target = np.array(self.target_pos)

        target_vel = (curr_target - prev_target) / (.001) #need clarification about dt

        joint_positions, joint_velocities = self.get_joint_positions_and_velocities()

        p.stepSimulation()

        self.update_logs(joint_positions, joint_velocities, target_vel, distances)
        self.container.update_log()
        state = self.update_state(joint_positions, joint_velocities, target_vel, distances)

        return state, final_reward, done
#to_do:
# things to hold model down