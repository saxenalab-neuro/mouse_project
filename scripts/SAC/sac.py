import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .utils1 import soft_update, hard_update
from .model import GaussianPolicy, GaussianPolicyLSTM, QNetwork, DeterministicPolicy
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
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space=None).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, h_prev, c_prev, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0).unsqueeze(0)
        h_prev = h_prev.to(self.device)
        c_prev = c_prev.to(self.device)
        
        if evaluate is False:
            action, _, _, h_current, c_current, _ = self.policy.sample(state, h_prev, c_prev, sampling= True, len_seq= None)
        else:
            _, _, action, h_current, c_current = self.policy.sample(state, h_prev, c_prev, sampling= True, len_seq= None)
        return action.detach().cpu().numpy()[0], h_current.detach(), c_current.detach()

    def update_parameters(self, policy_memory, policy_batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, h_batch, c_batch, policy_state_batch = policy_memory.sample(batch_size=policy_batch_size)
        
        # policy_state_batch= policy_memory.sample(batch_size= policy_batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        h_batch = torch.FloatTensor(h_batch).to(self.device).permute(1, 0, 2)
        c_batch = torch.FloatTensor(c_batch).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, _, _ = self.policy.sample(next_state_batch.unsqueeze(1), h_batch, c_batch, sampling= True)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

#        pi, log_pi, _ = self.policy.sample(state_batch)
        # Update the policy network using the newly proposed method
        h0 = torch.zeros(size=(1, len(policy_state_batch), self.hidden_size)).to(self.device)
        c0 = torch.zeros(size=(1, len(policy_state_batch), self.hidden_size)).to(self.device)

        len_seq = list(map(len, policy_state_batch))
        policy_state_batch = torch.FloatTensor(pad_sequence(policy_state_batch, batch_first= True)).to(self.device)
        pi_action_bat, log_prob_bat, _, _, _, mask_seq  = self.policy.sample(policy_state_batch, h0, c0, sampling= False, len_seq= len_seq)

        #Now mask the policy_state_batch according to the mask seq
        policy_state_batch_pi= policy_state_batch.reshape(-1, policy_state_batch.size()[-1])[mask_seq]

        qf1_pi, qf2_pi = self.critic(policy_state_batch_pi, pi_action_bat)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss_1 = ((self.alpha * log_prob_bat) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        #Find the loss encouraging simpler dynamics of the LSTM

        # An attempt to use pytorch to find the loss function(turned out not to work as pytorch does not track gradients)
        # Sample the hidden weights of the RNN
        # J_lstm_w = self.policy.lstm.weight_hh_l0        #These weights would be of the size (hidden_dim, hidden_dim)

        # #Sample the output of the RNN for the policy_state_batch
        # lstm_out_r = self.policy.forward_for_simple_dynamics(policy_state_batch, h0, c0, sampling= False, len_seq= len_seq)
        # lstm_out_r = lstm_out_r.reshape(-1, lstm_out_r.size()[-1])[mask_seq]

        # lstm_out_x = torch.arctanh(lstm_out_r)
        # lstm_out_x = torch.tensor(lstm_out_x, requires_grad= True)
        # lstm_out_x.detach()
        # J_lstm_w.detach()

        # lstm_out_r = torch.tanh(lstm_out_x).unsqueeze(-1)

        # dt_dynamics = torch.matmul(J_lstm_w, lstm_out_r).squeeze(-1)
        # # print(dt_dynamics.shape)

        # for i_grad in range(dt_dynamics.size()[-1]):
        #     if lstm_out_x.grad != None:
        #         lstm_out_x.grad.zero_()

        #     grad_vec= torch.zeros_like(dt_dynamics)
        #     grad_vec[:, i_grad] = 1
            
        #     dt_dynamics.backward(grad_vec, retain_graph= True)

        #     # print(lstm_out_x.grad)

        #     if i_grad==0:
        #         jac_l2 = torch.norm(lstm_out_x.grad)**2
        #     else:
        #         jac_l2 += torch.norm(lstm_out_x.grad)**2

        # print(jac_l2)
        # print(policy_loss_1)

        #Find the loss encouraging the simpler dynamics of the RNN

        # Sample the hidden weights of the RNN
        J_lstm_w = self.policy.lstm.weight_hh_l0        #These weights would be of the size (hidden_dim, hidden_dim)

        #Sample the output of the RNN for the policy_state_batch
        lstm_out_r, _ = self.policy.forward_for_simple_dynamics(policy_state_batch, h0, c0, sampling= False, len_seq= len_seq)
        lstm_out_r = lstm_out_r.reshape(-1, lstm_out_r.size()[-1])[mask_seq]

        #Reshape the policy hidden weights vector
        J_lstm_w = J_lstm_w.unsqueeze(0).repeat(lstm_out_r.size()[0], 1, 1)
        lstm_out_r = 1 - torch.pow(lstm_out_r, 2)

        R_j = torch.mul(J_lstm_w, lstm_out_r.unsqueeze(-1))

        policy_loss_2 = torch.norm(R_j)**2

        #Find the loss encouraging the minimization of the firing rates for the linear and the RNN layer
        #Sample the output of the RNN for the policy_state_batch
        lstm_out_r, linear_out = self.policy.forward_for_simple_dynamics(policy_state_batch, h0, c0, sampling= False, len_seq= len_seq)
        lstm_out_r = lstm_out_r.reshape(-1, lstm_out_r.size()[-1])[mask_seq]
        linear_out = linear_out.reshape(-1, linear_out.size()[-1])[mask_seq]

        policy_loss_3 = torch.norm(lstm_out_r)**2 + torch.norm(linear_out)**2

        # print(policy_loss_3)

        #Find the loss encouraging the minimization of the input and output weights of the LSTM(RNN) and the layers downstream
        #and upstream of the LSTM
        #Sample the input weights of the RNN
        J_lstm_i = self.policy.lstm.weight_ih_l0
        J_in1 = self.policy.linear1.weight

        #Sample the output weights
        # J_out = self.policy.linear2.weight
        J_out1 = self.policy.mean_linear.weight
        J_out2 = self.policy.log_std_linear.weight

        policy_loss_4 = torch.norm(J_in1)**2 + torch.norm(J_lstm_i)**2 + torch.norm(J_out1)**2 + torch.norm(J_out2)**2

        # print(policy_loss_4)

        policy_loss= policy_loss_1 + 0.1*policy_loss_2 + 0.01*policy_loss_3 + 0.001*policy_loss_4

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

        return qf1_loss.item(), qf2_loss.item(), policy_loss_1.item(), policy_loss_2.item(), policy_loss_3.item(), policy_loss_4.item(), alpha_loss.item(), alpha_tlogs.item()

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

