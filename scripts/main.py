import pybullet as p
import numpy as np
import time
import argparse
import itertools
import scipy.io
import torch
import matplotlib.pyplot as plt

import farms_pylog as pylog
import model_utils as model_utils
from Mouse_RL_Environment import Mouse_Env
from SAC.replay_memory import PolicyReplayMemory
from SAC.sac import SAC

file_path = "model_utilities/mouse_fixed.sdf" # mouse model, body fixed except for right arm
pose_file = "model_utilities/right_forelimb_pose.yaml" # pose file for original pose
muscle_config_file = "model_utilities/right_forelimb.yaml" # muscle file for right arm

model_offset = (0.0, 0.0, .0475) #z position modified with global scaling

#ARM CONTROL
ctrl = [107, 108, 109, 110, 111, 113, 114]
#ctrl = [3, 4, 5, 6, 7, 10, 11]

###JOINT TO INDEX###
#RShoulder_rotation - 107
#RShoulder_adduction - 108
#RShoulder_flexion - 109
#RElbow_flexion - 110
#RElbow_supination - 111
#RWrist_adduction - 113
#RWrist_flexion - 114
#RMetacarpus1_flexion - 115, use link (carpus) for pos
 
### INTERPOLATION
def interpolate(orig_data):

    interpolated = []
    for i in range(len(orig_data)-1):
        interpolated.append(orig_data[i])
        interpolated_point = (i + (i+1)) / 2
        y = orig_data[i] + (interpolated_point - i) * ((orig_data[i+1]-orig_data[i])/((i+1)-i))
        interpolated.append(y)
    interpolated.append(orig_data[-1])

    return interpolated

def main():
    ### PARAMETERS ###
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='G',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--policy_batch_size', type=int, default=8, metavar='N',
                        help='batch size (default: 6)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 1000)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=0, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--policy_replay_size', type=int, default=500000, metavar='N',
                        help='size of replay buffer (default: 2800)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--visualize', type=bool, default=False,
                        help='visualize mouse')
    args = parser.parse_args()

    ###SIMULATION PARAMETERS###
    frame_skip = 1
    n_frames = 1
    timestep = 170

    ### CREATE ENVIRONMENT, AGENT, MEMORY ###
    mouseEnv = Mouse_Env(file_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset, args.visualize)
    agent = SAC(41, mouseEnv.action_space, args)
    policy_memory= PolicyReplayMemory(args.policy_replay_size, args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ### DISABLES CURRENT MOVEMENT ###
    model_utils.disable_control(mouseEnv.model)

    ### 1SEC REAL TIME = 1 ms SIMULATION ###
    p.setTimeStep(.001)

    ### INITIALIZE ALL VALUES TO TRACK ###
    total_numsteps = 0
    updates = 0
    score_history= []
    reward_tracker_slow = []
    reward_tracker_fast = []
    reward_tracker_1 = []
    test_history= []
    vel_his = []
    target_trajectory = []
    env_counter_his= []
    kinematics_x = []
    kinematics_y = []
    kinematics_z = []
    reward_tracker = []
    policy_loss_tracker = []
    highest_reward = 0

    ### DATA SET LOADING/PROCESSING ###
    dataset = ['data_fast', 'data_slow', 'data_1']

    ########################### Data_Fast ###############################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt_fast.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_fast = data[231:401:1] * -1
    data_fast = [-13.45250312, *data_fast[8:]]
    # Data must start and end at same spot or there is jump
    mouse_fast = np.zeros_like(data_fast)
    data_fast_avg = 0
    data_fast_rewards = [0]

    ########################### Data_Slow ###############################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt_slow.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_slow = data[256:476:1] * -1
    mouse_slow = np.zeros_like(data_slow)
    data_slow_avg = 0
    data_slow_rewards = [0]

    ############################ Data_1 ##############################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt1.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_1= data[226:406:1] * -1
    data_1 = [-13.45250312, *data_1[4:]]
    mouse_1 = np.zeros_like(data_1)
    data_1_avg = 0
    data_1_rewards = [0]

    data_fast_pos = 1
    data_slow_pos = 1
    data_1_pos = 1

    ### BEGIN TRAINING LOOP
    for i_episode in itertools.count(1):
        ### INITALIZE SIM PARAMETERS ###
        episode_reward = 0
        episode_steps = 0
        action_list= []
        done = False

        ### DATA SELECTION BY AVERAGE PERFORMANCE ###
        if i_episode % 3 == 0:
            mouseEnv.timestep = len(data_fast)
            mouseEnv.x_pos = data_fast
            data_curr = dataset[0]
        elif i_episode % 3 == 1:
            mouseEnv.timestep =  len(data_slow)
            mouseEnv.x_pos = data_slow
            data_curr = dataset[1]
        elif i_episode % 3 == 2:
            mouseEnv.timestep = len(data_1)
            mouseEnv.x_pos = data_1
            data_curr = dataset[2]
 
        ### GET INITAL STATE + RESET MODEL BY POSE
        mouseEnv.reset(pose_file)
        state = mouseEnv.get_cur_state()
        ep_trajectory = []

        #num_layers specified in the policy model 
        h_prev = torch.zeros(size=(1, 1, args.hidden_size))
        c_prev = torch.zeros(size=(1, 1, args.hidden_size))

        ### STEPS PER EPISODE ###
        for i in range(mouseEnv.timestep):
            with torch.no_grad():
                if args.start_steps > total_numsteps:
                    action = mouseEnv.action_space.sample()  # Sample random action
                else:
                    action, h_current, c_current = agent.select_action(state, h_prev, c_prev)  # Sample action from policy

            action_list.append(action)
            
            ### SIMULATION ###
            if len(policy_memory.buffer) > args.policy_batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(policy_memory, args.policy_batch_size, updates)
                    policy_loss_tracker.append(policy_loss)
                    updates += 1

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            next_state, reward, done = mouseEnv.step(action, i_episode)
            episode_reward += reward

            mask = 1 if episode_steps == mouseEnv._max_episode_steps else float(not done)

            if episode_steps == 0:
                ep_trajectory.append((state, action, np.array([reward]), next_state, np.array([mask]), h_prev, c_prev, h_current, c_current))
            else:
                ep_trajectory.append((state, action, np.array([reward]), next_state, np.array([mask])))

            state = next_state
            h_prev = h_current
            c_prev = c_current

            episode_steps += 1
            total_numsteps += 1 
            
            ### EARLY TERMINATION OF EPISODE
            if done:
                break
        
        ### SAVING MODELS + TRACKING VARIABLES ###
        if episode_reward > highest_reward:
            highest_reward = episode_reward 

            #pylog.debug("Saving policy and Q network")
            #torch.save(agent.policy.state_dict(), 'models/policy_net_best.pth')
            #torch.save(agent.critic.state_dict(), 'models/value_net_best.pth')
        
        #torch.save(agent.policy.state_dict(), 'models/policy_net_cur.pth')
        #torch.save(agent.critic.state_dict(), 'models/value_net_cur.pth')

        pylog.debug('Iteration: {} | reward with total timestep {}: {}'.format(i_episode, mouseEnv.timestep, episode_reward))
        pylog.debug('highest reward so far: {}'.format(highest_reward))

        reward_tracker.append(episode_reward)
        policy_memory.push(ep_trajectory)

        np.savetxt('Score/rewards.txt', reward_tracker)
        np.savetxt('Score/policy_losses.txt', policy_loss_tracker)

    mouseEnv.close() #disconnects server

if __name__ == '__main__':
    main()
