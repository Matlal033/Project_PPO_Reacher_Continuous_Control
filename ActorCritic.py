import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#State size is 33
#Action size is 4
#Value size is 1
class MainBody(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, LR):
        super(MainBody, self).__init__()

        self.actor = nn.Sequential(
             nn.Linear(input_size, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        mb = self.actor(x)
        return mb

class ActorCritic(nn.Module):

    def __init__(self, state_size, action_size, hidden_size, LR, adam_epsilon, std=0.0):
        super(ActorCritic, self).__init__()

        self.actor = MainBody(state_size, action_size, hidden_size, LR)
        self.critic = MainBody(state_size, 1, hidden_size, LR)
        self.optimizer = optim.Adam(self.parameters(), lr=LR, eps=adam_epsilon)

        self.std = nn.Parameter(torch.ones(1, action_size))

    def forward(self, x):
        mu = torch.tanh(self.actor(x)) #output limited to range ]-1,1[, but because of std, we will need to clip actions after also
        a_dist = torch.distributions.Normal(mu, self.std)

        values = self.critic(x).squeeze(-1)

        return a_dist, values
