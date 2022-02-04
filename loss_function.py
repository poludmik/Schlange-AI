import torch
import torch.nn as nn
import model

DONE = 666

class BellmanLoss(nn.Module):

    def __init__(self, Qnet_model: model.Qnet, discount_factor=0.9):
        super(BellmanLoss, self).__init__()
        self.discount_factor = discount_factor
        self.model = Qnet_model


    def compute_loss(self, transition: list):

        lossF = nn.MSELoss(reduction='sum')
        # lossF = nn.HuberLoss(reduction='sum')

        state0 = transition[0]
        action = transition[1]
        reward = transition[2]
        state1 = transition[3]
        Q_values=transition[4]

        Q_new = torch.clone(Q_values)

        if state1[0] == DONE:
            Q_new[torch.argmax(Q_values).item()] = reward
        else:
            Q_new[torch.argmax(Q_values).item()] = reward + self.discount_factor * torch.max(self.model.forward_pass(state1))

        #print("Q_0:", Q_values)
        #print("Q_n:", Q_new)

        return lossF(Q_values, Q_new)

