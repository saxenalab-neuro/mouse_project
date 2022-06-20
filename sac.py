import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from utils1 import soft_update, hard_update
from model import GaussianPolicy, GaussianPolicyLSTM, QNetwork, DeterministicPolicy
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.hidden_size= args.hidden_size

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicyLSTM(num_inputs, action_space.shape[0], args.hidden_size, action_space=None).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, h_prev, c_prev, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0).unsqueeze(0)
        h_prev = h_prev.to(self.device)
        c_prev = c_prev.to(self.device)
        
        if evaluate is False:
            action, _, _, h_current, c_current = self.policy.sample(state, h_prev, c_prev, sampling= True)
        else:
            _, _, action, h_current, c_current = self.policy.sample(state, h_prev, c_prev, sampling= True)
        return action.detach().cpu().numpy()[0], h_current.detach(), c_current.detach()

    def update_parameters(self, policy_memory, policy_batch_size, updates):
        # Sample a batch from memory
        #state_batch_p means padded_batch state_batch1 in notes
        #state_batch means packed batch state_batch in notes

        state_batch_0, action_batch_0, reward_batch_0, next_state_batch_0, mask_batch_0, hidden_in, hidden_out = policy_memory.sample(batch_size=policy_batch_size)

        seq_lengths= list(map(len, state_batch_0))

        state_batch_p = pad_sequence(state_batch_0, batch_first= True)
        action_batch_p = pad_sequence(action_batch_0, batch_first= True)
        reward_batch_p = pad_sequence(reward_batch_0, batch_first= True)
        next_state_batch_p = pad_sequence(next_state_batch_0, batch_first= True)
        mask_batch_p = pad_sequence(mask_batch_0, batch_first= True)

        state_batch_p = torch.FloatTensor(state_batch_p).to(self.device)
        next_state_batch_p = torch.FloatTensor(next_state_batch_p).to(self.device)
        action_batch_p = torch.FloatTensor(action_batch_p).to(self.device)
        reward_batch_p = torch.FloatTensor(reward_batch_p).to(self.device)
        mask_batch_p = torch.FloatTensor(mask_batch_p).to(self.device)
        hidden_in = (hidden_in[0].to(self.device), hidden_in[1].to(self.device))
        hidden_out = (hidden_out[0].to(self.device), hidden_out[1].to(self.device))

        state_batch = pack_padded_sequence(state_batch_p, seq_lengths, batch_first= True, enforce_sorted= False)
        next_state_batch = pack_padded_sequence(next_state_batch_p, seq_lengths, batch_first= True, enforce_sorted= False)
        action_batch = pack_padded_sequence(action_batch_p, seq_lengths, batch_first= True, enforce_sorted= False)
        reward_batch_pack = pack_padded_sequence(reward_batch_p, seq_lengths, batch_first= True, enforce_sorted= False)
        mask_batch_pack = pack_padded_sequence(mask_batch_p, seq_lengths, batch_first= True, enforce_sorted= False)

        reward_batch = self.filter_padded(reward_batch_p, seq_lengths)
        mask_batch = self.filter_padded(mask_batch_p, seq_lengths)

        # We have padded batches of state, action, reward, next_state and mask from here downwards. We also have corresponding sequence lengths seq_lens
        # batch_p stands for padded batch or tensor of size (B, L_max, H)
        with torch.no_grad():

            next_state_action_p, next_state_log_pi_p, _, _, _ = self.policy.sample(next_state_batch, h_prev=hidden_out[0], c_prev= hidden_out[1], sampling= False)
            next_state_state_action_p = torch.cat((next_state_batch_p, next_state_action_p), dim=2)
            next_state_state_action = pack_padded_sequence(next_state_state_action_p, seq_lengths, batch_first= True, enforce_sorted= False)

            qf1_next_target, qf2_next_target = self.critic_target(next_state_state_action, hidden_out)

            qf1_next_target = self.filter_padded(qf1_next_target, seq_lengths)
            qf2_next_target = self.filter_padded(qf2_next_target, seq_lengths)
            next_state_log_pi = self.filter_padded(next_state_log_pi_p, seq_lengths)

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        
        state_action_batch_p = torch.cat((state_batch_p, action_batch_p), dim=2)
        state_action_batch = pack_padded_sequence(state_action_batch_p, seq_lengths, batch_first= True, enforce_sorted= False)

        qf1_p, qf2_p = self.critic(state_action_batch, hidden_in)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = self.filter_padded(qf1_p, seq_lengths)
        qf2 = self.filter_padded(qf2_p, seq_lengths)

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Update the policy network using the newly proposed method

        pi_action_bat_p, log_prob_bat_p, _, _, _ = self.policy.sample(state_batch, h_prev= hidden_in[0], c_prev= hidden_in[1], sampling= False)

        pi_state_action_batch_p = torch.cat((state_batch_p, pi_action_bat_p), dim=2)
        pi_state_action_batch = pack_padded_sequence(pi_state_action_batch_p, seq_lengths, batch_first= True, enforce_sorted= False)


        qf1_pi_p, qf2_pi_p = self.critic(pi_state_action_batch, hidden_in)
        qf1_pi = self.filter_padded(qf1_pi_p, seq_lengths)
        qf2_pi = self.filter_padded(qf2_pi_p, seq_lengths)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        log_prob_bat = self.filter_padded(log_prob_bat_p, seq_lengths)

        policy_loss = ((self.alpha * log_prob_bat) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        log_pi = log_prob_bat
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

    #filter_padded takes in a padded sequence of size (B, L_max, H) and corresponding sequence lengths, and returns a tensor of size [max(seq_lens), H]
    #after filtering redundant paddings. 
    def filter_padded(self, padded_seq, seq_lens):

        #   padded_seq = a tensor of size (batch_size, max_seq_len, input_dimension) i.e. (B, L_max, H) representing a padded object
        #   seq_lens = a list contatining the length of individual sequences in the sequence object before padding

        seq_max = max(seq_lens)

        #reshape padded sequence to (B*L_max, input_dimension)

        t = padded_seq.reshape(padded_seq.shape[0]*padded_seq.shape[1], padded_seq.shape[2])

        iter_max = int(t.shape[0]/seq_max)

        for iter1 in range(iter_max):
            k = [item for item in range(iter1*seq_max, (iter1+1)*seq_max)]
            k = k[:seq_lens[iter1]]

            if iter1 == 0:
                out_t = t[k]
            else:
                out_t = torch.cat((out_t, t[k]), dim=0)

        return out_t