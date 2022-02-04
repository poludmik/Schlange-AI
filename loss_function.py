import copy

import torch
import torch.nn as nn
import model
import experience_stack
import numpy as np

DONE = 666

class BellmanLoss(nn.Module):

    def __init__(self, Qnet_model: model.Qnet, discount_factor=0.9):
        super(BellmanLoss, self).__init__()
        self.discount_factor = discount_factor
        self.model = Qnet_model

    def compute_loss(self, batch, current_model, target_model):

        lossF = nn.MSELoss(reduction='sum')
        # lossF = nn.HuberLoss(reduction='sum')

        states0, actions, rewards, states1 = zip(*batch)

        # print("actions in loss:", actions)

        states0 = torch.tensor(states0, dtype=torch.float)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        states1 = torch.tensor(states1, dtype=torch.float)

        Q_values = self.model.forward_pass(states0)
        Q_new = torch.clone(Q_values)

        for idx in range(len(states0)):
            if states1[idx][0] == DONE:
                Q_new[idx][torch.argmax(actions[idx])] = rewards[idx]
            else:
                Q_new[idx][torch.argmax(actions[idx])] = rewards[idx] + self.discount_factor * torch.max(self.model.forward_pass(states1[idx]))

        return lossF(Q_values, Q_new)

