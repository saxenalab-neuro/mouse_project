import gym
import numpy as np
import model_utils
import pybullet as p
import pybullet_data

#FOR MUSCLES
#import farms_pylog as pylog
#try:
#    from farms_muscle.musculo_skeletal_system import MusculoSkeletalSystem
#except ImportError:
#    pylog.warning("farms-muscle not installed!")
#from farms_container import Container

model_offset = (0.0, 0.0, 1.2) #z position modified with global scaling

class PyBulletEnv(gym.Env):
    def __init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep):
        #####BUILDS SERVER AND LOADS MODEL#####
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8) #normal gravity
        self.plane = p.loadURDF("plane.urdf") #sets floor
        self.model = p.loadSDF(model_path, globalScaling = 25)[0]#resizes, loads model, returns model id
        p.resetBasePositionAndOrientation(self.model, model_offset, p.getQuaternionFromEuler([0, 0, 80.2])) #resets model position

        self.ctrl = ctrl
        self.pose_file = pose_file
        
        #####MUSCLES + DATA LOGGING#####
        #self.container = Container(max_iterations=int(2.5/0.001))

        # Physics simulation to namespace
        #self.sim_data = self.container.add_namespace('physics')
        #self.create_tables()

        #self.muscles = MusculoSkeletalSystem(self.container, 1e-3, config_path=muscle_config_file)

        #model_utils.reset_model_position(self.model, self.pose_file)
        
        #self.container.initialize()
        #self.muscles.print_system() 
        #print("num states", self.muscles.muscle_sys.num_states)
        #self.muscles.setup_integrator()

        #####META PARAMETERS FOR SIMULATION#####
        self.n_fixedsteps= 20
        self.timestep_limit= (1319 * 1) + self.n_fixedsteps
        self._max_episode_steps= 1000 #Does not matter. It is being set in the main.py where the total number of steps are being changed.
        self.threshold_user= 0.064
        self.timestep = timestep
        self.frame_skip= frame_skip


        self.stability_pos = p.getLinkState(self.model, 12)[0]

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
        self.reset_model(pose_file)

    def do_simulation(self, n_frames, forcesArray):
        for _ in range(n_frames):
            p.setJointMotorControlArray(self.model, self.ctrl, p.TORQUE_CONTROL, forces = forcesArray)

    #####DISCONNECTS SERVER#####
    def close(self):
        p.disconnect(self.client)

    def create_tables(self):
        self.sim_data.add_table('joint_positions')
        self.sim_data.add_table('joint_velocity')
        self.sim_data.add_table('distances')

        self.sim_data.joint_positions.add_parameter('RShoulder_rotation')
        self.sim_data.joint_positions.add_parameter('RShoulder_adduction')
        self.sim_data.joint_positions.add_parameter('RShoulder_flexion')
        self.sim_data.joint_positions.add_parameter('RElbow_flexion')
        self.sim_data.joint_positions.add_parameter('RElbow_supination')
        self.sim_data.joint_positions.add_parameter('RWrist_adduction')
        self.sim_data.joint_positions.add_parameter('RWrist_flexion')
        self.sim_data.joint_positions.add_parameter('RKnee')
        self.sim_data.joint_positions.add_parameter('LKnee')
        self.sim_data.joint_positions.add_parameter('LWrist_adduction')
        self.sim_data.joint_positions.add_parameter('LWrist_flexion')

        self.sim_data.joint_velocity.add_parameter('RShoulder_rotation')
        self.sim_data.joint_velocity.add_parameter('RShoulder_adduction')
        self.sim_data.joint_velocity.add_parameter('RShoulder_flexion')
        self.sim_data.joint_velocity.add_parameter('RElbow_flexion')
        self.sim_data.joint_velocity.add_parameter('RElbow_supination')
        self.sim_data.joint_velocity.add_parameter('RWrist_adduction')
        self.sim_data.joint_velocity.add_parameter('Rwrist_flexion')
        self.sim_data.joint_velocity.add_parameter('RKnee')
        self.sim_data.joint_velocity.add_parameter('LKnee')
        self.sim_data.joint_velocity.add_parameter('LWrist_adduction')
        self.sim_data.joint_velocity.add_parameter('LWrist_flexion')

        self.sim_data.distances.add_parameter('x')
        self.sim_data.distances.add_parameter('y')
        self.sim_data.distances.add_parameter('z')


    def update_logs(self, joint_positions, joint_velocities, distances):
        self.sim_data.joint_positions.values = np.asarray(joint_positions)
        self.sim_data.joint_velocity.values = np.asarray(joint_velocities)
        self.sim_data.distances.values = np.asarray(distances)

class Mouse_Stability_Env(PyBulletEnv):

    def __init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep):
        PyBulletEnv.__init__(self, model_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep)
        #u = self.container.muscles.activations
        #for muscle in self.muscles.muscles.keys():
        #       self.muscle_params[muscle] = u.get_parameter('stim_{}'.format(muscle))
        #       self.muscle_excitation[muscle] = p.addUserDebugParameter("flexor {}".format(muscle), 0, 1, 0.00)
        #       self.muscle_params[muscle].value = 0

    def reset_model(self, pose_file): 
        model_utils.reset_model_position(self.model, pose_file)

    def get_reward(self):  
        lumbar_curr_pos = p.getLinkState(self.model, 112)[0]
        d_x = np.abs(lumbar_curr_pos[0] - self.stability_pos[0])
        d_y = np.abs(lumbar_curr_pos[1] - self.stability_pos[1])
        d_z = np.abs(lumbar_curr_pos[2] - self.stability_pos[2])

        punishment = (-5 * d_x) - (5 * d_y) - (5 * d_z)

        reward = 5 * punishment 

        distances = [d_x, d_y, d_z]

        if np.abs(punishment)> 5:
            reward = -5

        return reward, distances

    def get_joint_positions_and_velocities(self):
        joint_positions = []
        joint_velocities = []
        for i in range(len(self.ctrl)):
            joint_positions.append(p.getJointState(self.model, self.ctrl[i])[0])
            joint_velocities.append(p.getJointState(self.model, self.ctrl[i])[1])
        return joint_positions, joint_velocities

    def is_done(self):
        if self.istep > self.timestep_limit:
            return True

        return False

    def step(self, forces):
        self.istep += 1

        self.do_simulation(self.frame_skip, forces)
        #self.muscles.step()

        reward, distances = self.get_reward()
        final_reward= (5*reward)

        done= self.is_done()

        joint_positions, joint_velocities = self.get_joint_positions_and_velocities()

        p.stepSimulation()

        #self.update_logs(joint_positions, joint_velocities, distances)
        #self.container.update_log()

        state = list(joint_positions) #joint positions
        state.append(joint_velocities) #joint velocities
        state.append(distances)# lumbar movement

        
        return state, final_reward, done
