import torch
import torch.nn as nn


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()

        self.linear_input = nn.Linear(11, 512, bias=True)
        self.BNi = nn.BatchNorm1d(24)

        self.hidden_layer1 = nn.Linear(512, 256, bias=True)
        self.BN1 = nn.BatchNorm1d(24)

        self.output_layer = nn.Linear(512, 3, bias=True)
        self.BNo = nn.BatchNorm1d(24)

        self.relu = nn.LeakyReLU(0.1)


    def forward_pass(self, state):

        state = self.linear_input(state)
        # state = self.BNi(state)
        # state = self.relu(state)

        # state = self.hidden_layer1(state)
        # state = self.BN1(state)
        # state = self.relu(state)

        state = self.output_layer(state)
        # state = self.BNo(state)
        state = self.relu(state)

        Q_values = state

        return Q_values



    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)



