import pybullet as p
import numpy as np
import time
import torch
import argparse
from sac import SAC
import itertools

import model_utils
from Mouse_RL_Environment import Mouse_Env
from Mouse_Stabilize_Environment import Mouse_Stability_Env
from replay_memory import ReplayMemory, PolicyReplayMemory

file_path = "./files/mouse_fixed.sdf" ###fixed mouse, arm training
#file_path = "/files/mouse_test.sdf" ###test mouse, stability training
pose_file = "./files/default_pose.yaml"
muscle_config_file = "./files/right_forelimb.yaml"

model_offset = (0.0, 0.0, .0475) #z position modified with global scaling

#ARM CONTROL
ctrl = [104, 105, 106, 107, 108, 110, 111]

#STABILITY CONTROL
#ctrl = [142, 125, 91, 92, 104, 105, 106, 107, 108, 110, 111]

###JOINT TO INDEX###
#RKnee - 142
#LKnee - 125
#LWrist_adduction - 91
#LWrist_flexion - 92
#RShoulder_rotation - 104
#RShoulder_adduction - 105
#RShoulder_flexion - 106
#RElbow_flexion - 107
#RElbow_supination - 108
#RWrist_adduction - 110
#RWrist_flexion - 111
#RMetacarpus1_flextion - 112, use link (carpus) for pos
#Lumbar2_bending - 12, use link(lumbar 1) for stability reward

if __name__ == "__main__":

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
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--policy_batch_size', type=int, default=6, metavar='N',
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
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--policy_replay_size', type=int, default=4000, metavar='N',
                        help='size of replay buffer (default: 2800)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    args = parser.parse_args()

    ###PARAMETERS###
    frame_skip = 1
    n_frames = 1
    timestep = 1000
    mouseEnv = Mouse_Env(file_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset)
    # hard code num_inputs, 
    agent = SAC(23, mouseEnv.action_space, args)
    policy_memory= PolicyReplayMemory(args.policy_replay_size, args.seed)

    #STABILITY ENV
    #mouseEnv = Mouse_Stability_Env(file_path, muscle_config_file, frame_skip, ctrl, timestep)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_utils.disable_control(mouseEnv.model)
    p.setTimeStep(.001)

    # Training Loop
    total_numsteps = 0
    updates = 0
    score_history= []
    test_history= []
    vel_his = []
    target_trajectory = []
    env_counter_his= []

    for i_episode in itertools.count(1):

        episode_reward = 0
        episode_steps = 0
        action_list= []
        done = False

        mouseEnv.reset(pose_file)
        state = mouseEnv.get_cur_state()

        ep_trajectory = []

        #num_layers specified in the policy model 
        h_prev = torch.zeros(size=(1, 1, args.hidden_size))
        c_prev = torch.zeros(size=(1, 1, args.hidden_size))

        for i in range(timestep):

            if args.start_steps > total_numsteps:
                action = mouseEnv.action_space.sample()  # Sample random action
            else:
                action, h_current, c_current = agent.select_action(state, h_prev, c_prev)  # Sample action from policy

            action_list.append(action)

            if len(policy_memory) > args.policy_batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(policy_memory, args.policy_batch_size, updates)

                    # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    # writer.add_scalar('loss/policy', policy_loss, updates)
                    # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done = mouseEnv.step(action)
            print(done)

            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == mouseEnv._max_episode_steps else float(not done)

            if episode_steps == 0:
                ep_trajectory.append((state, action, np.array([reward]), next_state, np.array([mask]), h_prev, c_prev, h_current, c_current))
            else:
                ep_trajectory.append((state, action, np.array([reward]), next_state, np.array([mask])))

            if done:
                break

            state = next_state
            h_prev = h_current
            c_prev = c_current

            episode_steps += 1
            total_numsteps += 1 
        
        policy_memory.push(ep_trajectory)

        if total_numsteps > args.num_steps:
            break

    mouseEnv.close() #disconnects server

