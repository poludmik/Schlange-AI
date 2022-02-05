import torch
import torch.nn as nn
import model

DONE = 666 # Constant to indicate the end of an episode.

class BellmanLoss(nn.Module):
    """
    Loss function is defined according to the Bellman equation. Penalises bad Q value predictions from the main net
    using the target net, which is updated in the training_agent main loop every 100 episodes.
    """

    def __init__(self, Qnet_model: model.Qnet, target_model: model.Qnet, discount_factor=0.9):
        super(BellmanLoss, self).__init__()
        self.discount_factor = discount_factor
        self.model = Qnet_model
        self.target_model = target_model

    def compute_loss(self, batch):

        lossF = nn.MSELoss(reduction='sum')

        states0, actions, rewards, states1 = zip(*batch)

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

