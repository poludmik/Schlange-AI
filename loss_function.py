import torch
import torch.nn as nn
import model

DONE = 666

class BellmanLoss(nn.Module):

    def __init__(self, Qnet_model: model.Qnet, discount_factor=0.9):
        super(BellmanLoss, self).__init__()
        self.discount_factor = discount_factor
        self.model = Qnet_model


    def compute_loss(self, Q_values, transition: list):

        lossF = nn.MSELoss(reduction='sum')

        state0 = transition[0]
        action = transition[1]
        reward = transition[2]
        state1 = transition[3]

        Q = torch.max(Q_values)

        if state1 == DONE:
            Q_new = reward
        else:
            Q_new = reward + self.discount_factor * torch.max(self.model.forward_pass(state1))

        print("Dimension of Q_new:", Q_new.shape())

        return lossF(Q, Q_new)

