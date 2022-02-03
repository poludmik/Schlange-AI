import torch
import torch.nn as nn

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()

        self.linear_input = nn.Linear(11, 24, bias=True)

        self.hidden_layer1 = nn.Linear(24, 48, bias=True)

        self.hidden_layer2 = nn.Linear(48, 48, bias=True)

        self.output_layer = nn.Linear(48, 3, bias=True)

        self.relu = nn.LeakyReLU(0.1)

        self.sigmoid = nn.Sigmoid()


    def forward_pass(self, state):
        state = self.linear_input(state)
        state = self.relu(state)
        state = self.hidden_layer1(state)
        state = self.relu(state)
        # state = self.hidden_layer2(state)
        # state = self.relu(state)
        state = self.output_layer(state)
        state = self.sigmoid(state)
        Q_values = state
        print("Dimenstions of Q_values:", Q_values.shape())
        return Q_values



    def weight_init(self):
        for lay in self.modules():
            if type(lay) in [torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)



