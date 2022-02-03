import snake
import game_class
import matplotlib.pyplot as plt
import pygame
import random
import torch
import model
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

    def get_one_transition(self):
        self.game.ensure_game_is_running()
        state0 = torch.FloatTensor(self.game.get_current_state())

        # decide on action
        action = random.choice([LEFT, RIGHT, STRAIGHT])

        reward = self.game.play_given_action_for_learning(action=action)
        if reward == NEG_REWARD:
            state1 = DONE
        else:
            state1 = torch.FloatTensor(self.game.get_current_state())

        pygame.time.delay(100) # TODO, delete probably

        transition = [state0, action, reward, state1]
        # store transition to memory..?:)

        return transition


    def train(self, model_path=None, epochs=100):
        epochs = epochs
        if not torch.cuda.is_available():
            print("Cuda is not available.")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        learning_rate = 0.001
        batch_size = 8

        trained_model = model.Qnet()
        loss_fc = loss_function.BellmanLoss(trained_model, discount_factor=0.9)

        if model_path is not None:
            trained_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        trained_model.to(device)

        # TODO load dataset maybe

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        losses = []
        epochas = []
        plt.style.use('seaborn-whitegrid')
        for epoch in range(epochs):
            # train

            # TODO: do a forward pass

            # TODO: compute loss

            # TODO: do the backward pass and update the gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()



            epochas.append(epoch)


        
        return True



if __name__ == "__main__":
    print("Initiated learning process.")




