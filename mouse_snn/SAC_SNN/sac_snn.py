import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, RMSprop
from .utils1_snn import soft_update, hard_update
from .model_snn import CriticSNN, PolicySNN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.hidden_size= args.hidden_size
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda")

    def select_action(self, state, evaluate=False):
        pass

    def update_parametersRNN(self, policy_memory, policy_batch_size):
        pass

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

class SACSNN(SAC):
    def __init__(self, num_inputs, action_space, args):
        super(SACSNN, self).__init__(num_inputs, action_space, args)

        self.critic = CriticSNN(num_inputs+action_space, action_space, args.hidden_size).to(self.device)
        self.critic_target = CriticSNN(num_inputs+action_space, action_space, args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.policy = PolicySNN(num_inputs, action_space, args.hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, spks, mem, evaluate=False):

        state = torch.FloatTensor(state).to(self.device)

        if evaluate == False: 
            action, _, _, mem2_rec_next, spk2_rec_next = self.policy.sample(state, spks=spks, mem=mem, sampling=True)
        else:
            _, _, action, mem2_rec_next, spk2_rec_next = self.policy.sample(state, spks=spks, mem=mem, sampling=True)

        return action.detach().cpu().numpy()[0], mem2_rec_next, spk2_rec_next
    
    def init_leakys(self, network, recurrent=True):
        mem_dict = {}
        spk_dict = {}
        for name in network.named_children():
            if "lif" in name[0]:
                if recurrent == True:
                    spk_dict[name[0]], mem_dict[name[0]] = name[1].init_rleaky()
                else:
                    mem_dict[name[0]] = name[1].init_leaky()
        return mem_dict, spk_dict
    
    def update_step(self, optim, loss):
        optim.zero_grad()
        loss.backward()
        optim.step()

    def gather_dict_batch(self, orig_dict, batch_size):
        dict_batch = {}
        for key in orig_dict[0]:
            dict_batch[key] = []
            for i in range(batch_size):
                dict_batch[key].append(torch.FloatTensor(orig_dict[i][key]).to(self.device))
            dict_batch[key] = torch.stack(dict_batch[key]) 
        return dict_batch

    def update_parameters(self, policy_memory, policy_batch_size):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = policy_memory.sample(batch_size=policy_batch_size)

        state_batch = torch.stack(state_batch).to(self.device)
        next_state_batch = torch.stack(next_state_batch).to(self.device)
        action_batch = torch.stack(action_batch).to(self.device)
        reward_batch = torch.stack(reward_batch).to(self.device).unsqueeze(-1)
        mask_batch = torch.stack(mask_batch).to(self.device).unsqueeze(-1)

        # Gathering them in batches
        with torch.no_grad():
            policy_mems, policy_spks = self.init_leakys(self.policy, recurrent=True)
            next_state_action = torch.empty_like(action_batch).to(self.device)
            next_state_log_pi = torch.empty([policy_batch_size, 300, 1]).to(self.device)
            for i in range(next_state_batch.shape[1]):
                next_state_action[:, i, :], next_state_log_pi[:, i, :], _, policy_mems, policy_spks = self.policy.sample(next_state_batch[:, i, :].unsqueeze(1), spks=policy_spks, mem=policy_mems, sampling=False, training=True)

            # pass sequency through Q network
            critic_target_mems, critic_target_spks = self.init_leakys(self.critic_target, recurrent=True)

            qf1_next_target = torch.empty([policy_batch_size, 300, 1]).to(self.device)
            qf2_next_target = torch.empty([policy_batch_size, 300, 1]).to(self.device)
            for i in range(next_state_batch.shape[1]):
                qf1_next_target[:, i, :], qf2_next_target[:, i, :], critic_target_mems, critic_target_spks = self.critic_target(next_state_batch[:, i, :], next_state_action[:, i, :], spk=critic_target_spks, mem=critic_target_mems, training=True)

            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        critic_mems, critic_spks = self.init_leakys(self.critic, recurrent=True)

        qf1 = torch.empty([policy_batch_size, 300, 1]).to(self.device)
        qf2 = torch.empty([policy_batch_size, 300, 1]).to(self.device)
        for i in range(state_batch.shape[1]):
            qf1[:, i, :], qf2[:, i, :], critic_mems, critic_spks = self.critic(state_batch[:, i, :], action_batch[:, i, :], spk=critic_spks, mem=critic_mems, training=True)  # Two Q-functions to mitigate positive bias in the policy improvement step

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = (qf1_loss + qf2_loss)

        #self.update_step(self.critic_optim, qf_loss)
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        policy_mems, policy_spks = self.init_leakys(self.policy, recurrent=True)
        pi_action_bat = torch.empty_like(action_batch).to(self.device)
        log_prob_bat = torch.empty(policy_batch_size, 300, 1).to(self.device)
        for i in range(state_batch.shape[1]):
            pi_action_bat[:, i, :], log_prob_bat[:, i, :], _, policy_mems, policy_spks = self.policy.sample(state_batch[:, i, :].unsqueeze(1), spks=policy_spks, mem=policy_mems, sampling=False, training=True)

        critic_mems, critic_spks = self.init_leakys(self.critic, recurrent=True)

        qf1_pi = torch.empty([policy_batch_size, 300, 1]).to(self.device)
        qf2_pi = torch.empty([policy_batch_size, 300, 1]).to(self.device)
        for i in range(state_batch.shape[1]):
            qf1_pi[:, i, :], qf2_pi[:, i, :], critic_mems, critic_spks = self.critic(state_batch[:, i, :], pi_action_bat[:, i, :], spk=critic_spks, mem=critic_mems, training=True)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_prob_bat) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        print(policy_loss)

        #self.update_step(self.policy_optim, policy_loss)
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()
