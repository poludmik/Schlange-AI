import math

import snake
import game_class
import matplotlib.pyplot as plt
import pygame
import random
import torch
import numpy as np
import model
import torch.nn.functional as F
import loss_function

NEG_REWARD = -10
POS_REWARD = 10
LEFT = [0, 1, 0]
RIGHT = [0, 0, 1]
STRAIGHT = [1, 0, 0]
DONE = 666


class Agent:

    def __init__(self, size_of_gamefield=10):
        self.game = game_class.SnakeGame(size_of_gamefield, dark=True, window_size=400)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_one_transition(self, Qnet_model, expolation_probability=0.5):
        self.game.ensure_game_is_running()

        state0 = torch.FloatTensor(self.game.get_current_state())
        state0 = state0.to(self.device)

        # decide on action

        Q_values = Qnet_model.forward_pass(state0)

        if random.uniform(0, 1) < expolation_probability:
            action = random.choice([LEFT, RIGHT, STRAIGHT])
        else:
            action = [0, 0, 0]
            action[torch.argmax(Q_values).item()] = 1

        # print("\nAction:", action, type(action))

        reward = self.game.play_given_action_for_learning(action=action)

        if reward == NEG_REWARD:
            state1 = torch.FloatTensor(self.game.get_current_state())
            state1[0] = NEG_REWARD
        else:
            state1 = torch.FloatTensor(self.game.get_current_state())

        state1 = state1.to(self.device)

        # pygame.time.delay(100) # TODO, delete probably

        transition = [state0, action, reward, state1, Q_values]
        # store transition to memory..?:)

        return transition


    def train(self, model_path=None, epochs=100):
        epochs = epochs

        if not torch.cuda.is_available():
            print("Cuda is not available.")

        learning_rate = 0.01
        batch_size = 4

        trained_model = model.Qnet()
        loss_fc = loss_function.BellmanLoss(trained_model, discount_factor=0.9)

        torch.autograd.set_detect_anomaly(True)

        if model_path is not None:
            trained_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        trained_model.to(self.device)

        # TODO load dataset maybe

        optimizer = torch.optim.Adam(trained_model.parameters(), lr=learning_rate)

        losses = []
        epochas = []
        scores = []
        plt.style.use('seaborn-whitegrid')
        for epoch in range(epochs):
            # train

            # TODO: do a forward pass
            transition = self.get_one_transition(Qnet_model=trained_model)

            # TODO: compute loss
            loss = loss_fc.compute_loss(transition=transition)

            # TODO: do the backward pass and update the gradients
            if loss.item() > 10.0:
                print("LOSS sqrt (>0.1):", math.sqrt(loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # TODO: update target net

            epochas.append(epoch)
            losses.append(loss.item())
            scores.append(50 * len(self.game.current_snake.rest_of_body_positions))

            if (epoch + 1) % 75 == 0:
                plt.plot(epochas, losses, '-k', label='loss')
                #plt.plot(epochas, scores, '-b', label='score')

                # trend
                z = np.polyfit(epochas, losses, 1)
                p = np.poly1d(z)
                plt.plot(epochas, 10 * p(epochas), "b-")

                plt.show()

        plt.plot(epochas, losses, '-k', label='loss')

        # trend
        z = np.polyfit(epochas, losses, 1)
        p = np.poly1d(z)
        plt.plot(epochas, p(epochas), "b-")

        # plt.plot(epochas, scores, '-b', label='score')
        plt.show()

        return True



if __name__ == "__main__":
    print("Initiated learning process.")

    agent = Agent(size_of_gamefield=5)

    agent.train(model_path=None, epochs=10000)


