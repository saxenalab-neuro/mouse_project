import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# class QNetwork(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim):
#         super(QNetwork, self).__init__()

#         # Q1 architecture
#         self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear3 = nn.Linear(hidden_dim, 1)

#         # Q2 architecture
#         self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
#         self.linear5 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear6 = nn.Linear(hidden_dim, 1)

#         self.apply(weights_init_)

#     def forward(self, state, action):
#         xu = torch.cat([state, action], 1)
        
#         x1 = F.relu(self.linear1(xu))
#         x1 = F.relu(self.linear2(x1))
#         x1 = self.linear3(x1)

#         x2 = F.relu(self.linear4(xu))
#         x2 = F.relu(self.linear5(x2))
#         x2 = self.linear6(x2)

#         return x1, x2

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers= 1, batch_first= True)
        self.linear3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear5 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear6 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers= 1, batch_first= True)
        self.linear7 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)
        # notes: weights_init for the LSTM layer

    def forward(self, state_action_packed, hidden):

        xu = state_action_packed
        xu_p, seq_lens = pad_packed_sequence(xu, batch_first= True)

        fc_branch_1 = F.relu(self.linear1(xu_p))

        lstm_branch_1 = F.relu(self.linear2(xu_p))
        lstm_branch_1 = pack_padded_sequence(lstm_branch_1, seq_lens, batch_first= True, enforce_sorted= False)
        lstm_branch_1, hidden_out_1 = self.lstm1(lstm_branch_1, hidden)
        lstm_branch_1, _ = pad_packed_sequence(lstm_branch_1, batch_first= True)

        x1 = torch.cat([fc_branch_1, lstm_branch_1], dim=-1)
        x1 = F.relu(self.linear3(x1))
        x1 = F.relu(self.linear4(x1))


        fc_branch_2 = F.relu(self.linear5(xu_p))

        lstm_branch_2 = F.relu(self.linear6(xu_p))
        lstm_branch_2 = pack_padded_sequence(lstm_branch_2, seq_lens, batch_first= True, enforce_sorted= False)
        lstm_branch_2, hidden_out_2 = self.lstm2(lstm_branch_2, hidden)
        lstm_branch_2, _ = pad_packed_sequence(lstm_branch_2, batch_first= True)

        x2 = torch.cat([fc_branch_2, lstm_branch_2], dim=-1)
        x2 = F.relu(self.linear7(x2))
        x2 = F.relu(self.linear8(x2))

        return x1, x2
        

    # def forward(self, state_action_packed, hidden):
    #     xu = state_action_packed
        
    #     x1, (h_out, c_out) = self.lstm1(xu, hidden)
    #     #Unpack the output to pass it through the forward layer
    #     x1, seq_lens = pad_packed_sequence(x1, batch_first= True)
    #     x1 = F.relu(x1)
    #     x1 = F.relu(self.linear1(x1))

    #     x2, (h_out, c_out) = self.lstm2(xu, hidden)
    #     #Unpack the output to pass it through the forward layer
    #     x2, seq_lens = pad_packed_sequence(x2, batch_first= True)
    #     x2 = F.relu(x2)
    #     x2 = F.relu(self.linear1(x2))

    #     return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class GaussianPolicyLSTM(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicyLSTM, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.lstm = nn.LSTM(num_inputs, hidden_dim, num_layers= 1, batch_first= True)
        self.linear2 = nn.Linear(2*hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)
        # Adjust the initial weights of the recurrent LSTM layer

        # action rescaling
        # Pass none action space and adjust the action scale and bias manually
        if action_space is None:
            self.action_scale = torch.tensor(0.5)
            self.action_bias = torch.tensor(0.5)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, h_prev, c_prev, sampling):

        if sampling == True:
            fc_branch = F.relu(self.linear1(state))
            lstm_branch, (h_current, c_current) = self.lstm(state, (h_prev, c_prev))
        else:
            state_pad, _ = pad_packed_sequence(state, batch_first= True)
            fc_branch = F.relu(self.linear1(state_pad))
            lstm_branch, (h_current, c_current) = self.lstm(state, (h_prev, c_prev))
            lstm_branch, seq_lens = pad_packed_sequence(lstm_branch, batch_first= True)

        x = torch.cat([fc_branch, lstm_branch], dim=-1)
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std, h_current, c_current

    def sample(self, state, h_prev, c_prev, sampling):

        mean, log_std, h_current, c_current = self.forward(state, h_prev, c_prev, sampling)
        #if sampling == False; then reshape mean and log_std from (B, L_max, A) to (B*Lmax, A)

        mean_size = mean.size()
        log_std_size = log_std.size()

        mean = mean.reshape(-1, mean.size()[-1])
        log_std = log_std.reshape(-1, log_std.size()[-1])

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforce the action_bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        if sampling == False:
            action = action.reshape(mean_size[0], mean_size[1], mean_size[2])
            mean = mean.reshape(mean_size[0], mean_size[1], mean_size[2])
            log_prob = log_prob.reshape(log_std_size[0], log_std_size[1], 1) 

        return action, log_prob, mean, h_current, c_current
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyLSTM, self).to(device)



class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
