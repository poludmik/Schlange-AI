import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    """
    Small linear NN model to approximate the Q value function.
    """
    def __init__(self):
        super(Qnet, self).__init__()

        self.linear_input = nn.Linear(11, 256, bias=True)

        self.output_layer = nn.Linear(256, 3, bias=True)

    def forward_pass(self, state):

        state = self.linear_input(state)

        state = F.relu(state)

        state = self.output_layer(state)

        Q_values = state

        return Q_values
