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
from farms_container import DataTable

sphere_file = "/home/john_lazzari/mouse_project/files/sphere_small.urdf"
pose_file = "/home/john_lazzari/mouse_project/files/default_pose.yaml"

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
#Lumbar2_bending - 12, use link(lumbar 1) for stability reward

class PyBulletEnv(gym.Env):
    def __init__(self, model_path, muscle_config_file, frame_skip, ctrl, timestep):
        #####BUILDS SERVER AND LOADS MODEL#####
        #self.client = p.connect(p.GUI)
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8) #normal gravity
        self.plane = p.loadURDF("plane.urdf") #sets floor
        self.model = p.loadSDF(model_path, globalScaling = 25)[0]#resizes, loads model, returns model id
        p.resetBasePositionAndOrientation(self.model, model_offset, p.getQuaternionFromEuler([0, 0, 80.2])) #resets model position
        #self.sphere = p.loadURDF("sphere_small.urdf", globalScaling = 2) #visualizes target position

        self.ctrl = ctrl #control, list of all joints in right arm (shoulder, elbow, wrist + metacarpus for measuring hand pos)
        
        # Edited this section
        #####MUSCLES#####
        self.container = Container(max_iterations=int(2.5/0.001))

        # Physics simulation to namespace
        self.sim_data = self.container.add_namespace('physics')
        # Add tables to container
        self.sim_data.add_table('joint_positions')
        self.sim_data.add_table('joint_velocity')
        self.sim_data.add_table('target_positions')
        self.sim_data.add_table('target_velocity')
        self.sim_data.add_table('distances')

        self.muscles = MusculoSkeletalSystem(self.container, 1e-3, config_path=muscle_config_file)

        # Hardcoded rn
        self.sim_data.joint_positions.add_parameter('x')
        self.sim_data.joint_positions.add_parameter('y')
        self.sim_data.joint_positions.add_parameter('z')

        self.sim_data.joint_velocity.add_parameter('x')
        self.sim_data.joint_velocity.add_parameter('y')
        self.sim_data.joint_velocity.add_parameter('z')

        self.sim_data.target_positions.add_parameter('x')
        self.sim_data.target_positions.add_parameter('y')
        self.sim_data.target_positions.add_parameter('z')

        self.sim_data.target_velocity.add_parameter('x')
        self.sim_data.target_velocity.add_parameter('y')
        self.sim_data.target_velocity.add_parameter('z')

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
        self.sim_data.joint_velocity.add_parameter('rwrist_flexion')

        self.sim_data.target_positions.add_parameter('carpus')
        self.sim_data.target_velocity.add_parameter('carpus')

        model_utils.reset_model_position(self.model, pose_file)
        
        self.container.initialize()
        #self.muscles.print_system() 
        print("num states", self.muscles.muscle_sys.num_states)
        self.muscles.setup_integrator()

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
        self.center = [self.x_pos + .25 , self.y_pos, self.z_pos + .1]
        self.radius = .20
        self.theta = np.linspace(0, 2 * np.pi, self.timestep) #array from 0-2pi of timestep values
        self.stability_pos = p.getLinkState(self.model, 12)[0]
        #p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))

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
        #p.resetBasePositionAndOrientation(self.sphere, np.array(self.target_pos), p.getQuaternionFromEuler([0, 0, 80.2]))
        #print("x, y, z", self.target_pos)

    def step(self, forces):

        self.istep += 1

        #can edit threshold with episodes
        if self.istep > self.n_fixedsteps:
            self.threshold = 0.032

        self.do_simulation(self.frame_skip, forces)
        self.muscles.step()

        reward, distances = self.get_reward()
        cost = self.get_cost(forces)
        final_reward= (5*reward) - (0.5*cost)

        done= self.is_done()
        
        prev_target = np.array(self.target_pos)
        self.update_target_pos()
        curr_target = np.array(self.target_pos)

        target_vel = (curr_target - prev_target) / (self.frame_skip ) #need clarification about dt


        state = list(p.getJointStates(self.model, ctrl)[0]) #joint positions
        state.append(list(p.getJointStates(self.model, ctrl)[1])) #joint velocities
        state.append(list(self.target_pos)) #target position
        state.append(list(target_vel)) #target velocity
        state.append(distances)# hand_pos - target_pos

        return state, final_reward, done
#to_do:
# things to hold model down