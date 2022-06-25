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
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81) #normal gravity
        self.plane = p.loadURDF("plane.urdf") #sets floor
        self.model = p.loadSDF(model_path)[0]#resizes, loads model, returns model id
        self.model_offset = model_offset
        p.resetBasePositionAndOrientation(self.model, self.model_offset, p.getQuaternionFromEuler([0, 0, 80.2])) #resets model position
        self.use_sphere = False
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

        ####ADD TABLES TO CONTAINER###
        self.sim_data.add_table('joint_positions')
        self.sim_data.add_table('joint_velocity')
        self.sim_data.add_table('joint_torques')
        self.sim_data.add_table('base_position')

        #: Generate joint_name to id dict
        self.link_id[p.getBodyInfo(self.model)[0].decode('UTF-8')] = -1
        for n in range(228):
            info = p.getJointInfo(self.model, n)
            _id = info[0]
            joint_name = info[1].decode('UTF-8')
            link_name = info[12].decode('UTF-8')
            _type = info[2]
            self.joint_id[joint_name] = _id
            self.joint_type[joint_name] = _type
            self.link_id[link_name] = _id
            pylog.debug("Link name {} id {}".format(link_name, _id))

        #: ADD base position parameters
        self.sim_data.base_position.add_parameter('x')
        self.sim_data.base_position.add_parameter('y')
        self.sim_data.base_position.add_parameter('z')

        #: ADD joint paramters
        for name, _ in self.joint_id.items():
            print(name)
            self.sim_data.joint_positions.add_parameter(name)
            self.sim_data.joint_velocity.add_parameter(name)
            self.sim_data.joint_torques.add_parameter(name)

        self.initialize_muscles()
        model_utils.reset_model_position(self.model, self.pose_file)
        self.container.initialize()
        #self.muscles.setup_integrator()

        #####META PARAMETERS FOR SIMULATION#####
        self.n_fixedsteps= 10
        self.timestep_limit = timestep
        # self._max_episode_steps= self.timestep_limit/ 2
        self._max_episode_steps = timestep #Does not matter. It is being set in the main.py where the total number of steps are being changed.
        self.threshold_user= 0.01
        self.timestep = timestep
        self.frame_skip= frame_skip

        #####TARGET POSITION USING POINT IN SPACE: X, Y, Z#####
        ###x, y, z for initializing from hand starting position, target_pos for updating
        self.x_pos = p.getLinkState(self.model, 115)[0][0]
        self.y_pos = p.getLinkState(self.model, 115)[0][1]
        self.z_pos = p.getLinkState(self.model, 115)[0][2]

        self.radius = .006
        self.theta = np.linspace(7*np.pi/6, -5*np.pi/6, self.timestep) #array from 0-2pi of timestep values
        self.center = [self.x_pos + .0053, self.y_pos, self.z_pos + .005]
        self.target_pos = [self.radius * np.cos(self.theta[0]) + self.center[0], self.y_pos, self.radius * np.sin(self.theta[0]) + self.center[2]]
        if self.use_sphere:
            p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))

        p.resetDebugVisualizerCamera(0.3, 15, -10, [0, 0.21, 0])

        self.action_space = spaces.Box(low=np.ones(18), high=np.ones(18), dtype=np.float32)

        p.setJointMotorControlArray(
            self.model,
            self.ctrl,
            p.VELOCITY_CONTROL,
            targetVelocities=np.zeros(7),
            forces=np.zeros(7)
        )
        p.setJointMotorControlArray(
            self.model,
            self.ctrl,
            p.POSITION_CONTROL,
            forces=np.zeros(7)
        )
        p.setJointMotorControlArray(
            self.model,
            self.ctrl,
            p.TORQUE_CONTROL,
            forces=np.zeros(7)
        )

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
            p.setJointMotorControlArray(self.model, self.ctrl, p.TORQUE_CONTROL, forces=forcesArray)

    #####DISCONNECTS SERVER#####
    def close(self):
        p.disconnect(self.client)

    def create_tables(self):

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
    
    @property
    def base_position(self):
        """ Get the position of the animal  """
        return (p.getBasePositionAndOrientation(self.model))[0]

    @property
    def joint_positions(self):
        """ Get the joint positions in the animal  """
        return tuple(
            state[0] for state in p.getJointStates(
                self.model,
                np.arange(0, p.getNumJoints(self.model))
            )
        )

    @property
    def joint_torques(self):
        """ Get the joint torques in the animal  """
        return tuple(
            state[-1] for state in p.getJointStates(
                self.model,
                np.arange(0, p.getNumJoints(self.model))
            )
        )

    @property
    def joint_velocities(self):
        """ Get the joint velocities in the animal  """
        return tuple(
            state[1] for state in p.getJointStates(
                self.model,
                np.arange(0, p.getNumJoints(self.model))
            )
        )
    
    def initialize_muscles(self):
        self.muscles = MusculoSkeletalSystem(
            self.container, 1e-3, self.muscle_config_file
        )
   
    def update_logs(self):

        self.sim_data.joint_positions.values = np.asarray(self.joint_positions)
        self.sim_data.joint_velocity.values = np.asarray(self.joint_velocities)
        self.sim_data.joint_torques.values = np.asarray(self.joint_torques)
        self.sim_data.base_position.values = np.asarray(self.base_position)

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

        if d_x > self.threshold or d_y > self.threshold or d_z > self.threshold:
            reward = -5
        
        else:
            r_x= 1/(1000**d_x)
            r_y= 1/(1000**d_y)
            r_z= 1/(1000**d_z)

            reward= r_x + r_y + r_z

        return reward, distances

    def is_done(self):
        hand_pos =  np.array(p.getLinkState(self.model, 115)[0]) #(x, y, z)
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

    def update_state(self, act, joint_positions, joint_velocities, target_velocity, distances):
        state = [*list(act), *list(joint_positions), *list(joint_velocities), *list(self.target_pos), *list(target_velocity), *list(distances)]
        return state

    def get_cur_state(self):

        joint_positions, _ = self.get_joint_positions_and_velocities()
        _, distance = self.get_reward()
        return [*list(np.zeros(18)), *list(joint_positions), *[0., 0., 0., 0., 0., 0., 0.], *list(self.target_pos), *[0, 0, 0], *distance]
    
    def controller_to_actuator(self, forces):
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_AN", forces[0])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_BBL", forces[1])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_BBS", forces[2])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_BRA", forces[3])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_COR", forces[4])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_ECRB", forces[5])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_ECRL", forces[6])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_ECU", forces[7])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_EIP1", forces[8])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_EIP2", forces[9])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_FCR", forces[11])
        self.container.muscles.activations.set_parameter_value("stim_RIGHT_FORE_FCU", forces[10])
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

    def step(self, forces):

        # TODO
        # Make sure activations are correct and figure out importance of containers 
        # (seems to make different actions when doing container.update_log() but not sure why).
        # just changed containers to update logs and reset after each episode, seems to work much much better
        # change the state for muscle activations instead of joints

        self.istep += 1

        if self.istep == 1:
            #self.container.initialize()
            self.muscles.setup_integrator()

        self.controller_to_actuator(forces)
        #can edit threshold with episodes
        if self.istep > self.n_fixedsteps:
            self.threshold = 0.008

        #self.do_simulation(self.frame_skip, forces)
        self.muscles.step(forces, self.istep)
        act = self.get_activations()
        #self.update_logs()
        self.container.update_log()
        
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

        # Actually might be important for some reason
        state = self.update_state(act, joint_positions, joint_velocities, target_vel, distances)

        return state, final_reward, done
