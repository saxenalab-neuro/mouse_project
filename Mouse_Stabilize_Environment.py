import gym
import numpy as np
import model_utils
import pybullet as p
import pybullet_data
import yaml

#FOR MUSCLES
#import farms_pylog as pylog
#try:
#    from farms_muscle.musculo_skeletal_system import MusculoSkeletalSystem
#except ImportError:
#    pylog.warning("farms-muscle not installed!")
#from farms_container import Container

model_offset = (0.0, 0.0, 1.2) #z position modified with global scaling

class PyBulletEnv(gym.Env):
    def __init__(self, model_path, muscle_config_file, frame_skip, ctrl, timestep):
        #####BUILDS SERVER AND LOADS MODEL#####
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8) #normal gravity
        self.plane = p.loadURDF("plane.urdf") #sets floor
        self.model = p.loadSDF(model_path, globalScaling = 25)[0]#resizes, loads model, returns model id
        p.resetBasePositionAndOrientation(self.model, model_offset, p.getQuaternionFromEuler([0, 0, 80.2])) #resets model position

        self.ctrl = ctrl
        
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
            p.stepSimulation()

    #####DISCONNECTS SERVER#####
    def close(self):
        p.disconnect(self.client)

class Mouse_Stability_Env(PyBulletEnv):

    def __init__(self, model_path, muscle_config_file, frame_skip, ctrl, timestep):
        PyBulletEnv.__init__(self, model_path, muscle_config_file, frame_skip, ctrl, timestep)
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


        state = list(p.getJointStates(self.model, self.ctrl)[0]) #joint positions
        state.append(list(p.getJointStates(self.model, self.ctrl)[1])) #joint velocities
        state.append(distances)# lumbar movement

        
        return state, final_reward, done
