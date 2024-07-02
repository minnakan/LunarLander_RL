import torch
import torch.nn as nn
from math import pi as pi_constant
from typing import Tuple

Tensor = torch.DoubleTensor
torch.set_default_tensor_type(Tensor)


class SAC(nn.Module):
    def __init__(self, config):
        super(SAC, self).__init__()
        torch.manual_seed(config['seed'])

        self.lr = config['lr']  # learning rate
        self.smooth = config['smooth']  # smoothing coefficient for target net
        self.discount = config['discount']  # discount factor
        self.alpha = config['alpha']  # temperature parameter in SAC
        self.batch_size = config['batch_size']  # mini batch size

        self.dims_hidden_neurons = config['dims_hidden_neurons']
        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']

        self.device = config['device']

        self.actor = ActorNet(device=config['device'],
                              dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q1 = QCriticNet(dim_obs=self.dim_obs,
                             dim_action=self.dim_action,
                             dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q2 = QCriticNet(dim_obs=self.dim_obs,
                             dim_action=self.dim_action,
                             dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q1_tar = QCriticNet(dim_obs=self.dim_obs,
                                 dim_action=self.dim_action,
                                 dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q2_tar = QCriticNet(dim_obs=self.dim_obs,
                                 dim_action=self.dim_action,
                                 dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_Q1 = torch.optim.Adam(self.Q1.parameters(), lr=self.lr)
        self.optimizer_Q2 = torch.optim.Adam(self.Q2.parameters(), lr=self.lr)

    def update(self, buffer):
        # sample from replay memory
        t = buffer.sample(self.batch_size)
        states = t.obs.to(self.device)
        actions = t.action.to(self.device)
        rewards = t.reward.to(self.device)
        next_states = t.next_obs.to(self.device)
        dones = t.done.float().to(self.device)

        # TO DO: Perform the updates for the actor and critic networks
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor(next_states)
            q1_next_target = self.Q1_tar(next_states, next_state_action)
            q2_next_target = self.Q2_tar(next_states, next_state_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + (1.0 - dones) * self.discount * min_q_next_target

        q1_value = self.Q1(states, actions)
        q2_value = self.Q2(states, actions)
        q1_loss = nn.MSELoss()(q1_value, next_q_value)
        q2_loss = nn.MSELoss()(q2_value, next_q_value)

        self.optimizer_Q1.zero_grad()
        q1_loss.backward()
        self.optimizer_Q1.step()

        self.optimizer_Q2.zero_grad()
        q2_loss.backward()
        self.optimizer_Q2.step()

        # Update Actor network
        pi, log_pi, _ = self.actor(states)
        q1_pi = self.Q1(states, pi)
        q2_pi = self.Q2(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Update target networks
        for target_param, param in zip(self.Q1_tar.parameters(), self.Q1.parameters()):
            target_param.data.copy_(
                self.smooth * param.data + (1.0 - self.smooth) * target_param.data
            )
        for target_param, param in zip(self.Q2_tar.parameters(), self.Q2.parameters()):
            target_param.data.copy_(
                self.smooth * param.data + (1.0 - self.smooth) * target_param.data
            )

    def act_probabilistic(self, obs: torch.Tensor):
        self.actor.eval()
        a, logProb, mu = self.actor(obs)
        self.actor.train()
        return a

    def act_deterministic(self, obs: torch.Tensor):
        self.actor.eval()
        a, logProb, mu = self.actor(obs)
        self.actor.train()
        return mu


class ActorNet(nn.Module):
    def __init__(self,
                 device,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)
                 ):
        super(ActorNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action
        self.device = device

        self.ln2pi = torch.log(Tensor([2*pi_constant]))

        n_neurons = (dim_obs,) + dims_hidden_neurons + (dim_action,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output_mu = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output_mu.weight)
        torch.nn.init.zeros_(self.output_mu.bias)

        self.output_logsig = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output_logsig.weight)
        torch.nn.init.zeros_(self.output_logsig.bias)

    def forward(self, obs: torch.Tensor):
        x = obs
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        mu = self.output_mu(x)
        sig = torch.exp(self.output_logsig(x))

        # for the log probability under tanh-squashed Gaussian, see Appendix C of the SAC paper
        u = mu + sig * torch.normal(torch.zeros(size=mu.shape), 1).to(self.device)
        a = torch.tanh(u)
        logProbu = -1/2 * (torch.sum(torch.log(sig**2), dim=1, keepdims=True).to(self.device) +
                           torch.sum((u-mu)**2/sig**2, dim=1, keepdims=True) +
                           a.shape[1]*self.ln2pi.to(self.device))
        logProba = logProbu - torch.sum(torch.log(1 - a ** 2 + 0.000001), dim=1, keepdims=True)
        return a, logProba, torch.tanh(mu)


class QCriticNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        super(QCriticNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action

        n_neurons = (dim_obs + dim_action,) + dims_hidden_neurons + (1,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat((obs, action), dim=1)
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        return self.output(x)


