import copy
import math
import snake
import game_class
import matplotlib.pyplot as plt
import pygame
import random
import torch
import numpy as np
import pandas as pd
import model
import torch.nn.functional as F
import loss_function
import experience_stack


NEG_REWARD = -10
POS_REWARD = 10
LEFT = [0, 1, 0]
RIGHT = [0, 0, 1]
STRAIGHT = [1, 0, 0]
DONE = 666


def plot_info(plot, epochas, losses, scores):
    # plt.plot(epochas, losses, '-k', label='loss')
    # plt.plot(epochas, scores, '-b', label='score')

    plot.set_xdata(np.append(plot.get_xdata(), losses))
    plot.set_ydata(np.append(plot.get_ydata(), epochas))

    # trend
    df = pd.DataFrame(np.transpose(np.array([losses])))
    # plt.plot(df.expanding(min_periods=10).mean(), 'b')
    plt.draw()


class Agent:

    def __init__(self, size_of_gamefield=10, animate=False):
        self.game = game_class.SnakeGame(size_of_gamefield, dark=True, window_size=400, animate=animate)
        self.memory_stack = experience_stack.Memory()
        self.losses = []
        self.epochas = []
        self.scores = []

    def get_one_transition(self, Qnet_model, expolation_probability=0.5):
        self.game.ensure_game_is_running()

        state0 = self.game.get_current_state()

        Q_values = Qnet_model.forward_pass(torch.FloatTensor(state0))

        if random.uniform(0, 1) < expolation_probability:
            action = random.choice([LEFT, RIGHT, STRAIGHT])
        else:
            action = [0, 0, 0]
            action[torch.argmax(Q_values).item()] = 1

        reward = self.game.play_given_action_for_learning(action=action)

        if reward == NEG_REWARD:
            state1 = self.game.get_current_state()
            state1[0] = DONE
        else:
            state1 = self.game.get_current_state()

        transition = (state0, action, reward, state1)
        return transition


    def train(self, model_to_train, model_target, batch_size):

        # get a batch of transitions
        batch = random.sample(self.memory_stack.stack, batch_size)
        #batch.to(self.device)

        loss_fc = loss_function.BellmanLoss(model_to_train, discount_factor=0.9)

        # TODO: compute loss
        loss_local = loss_fc.compute_loss(batch, model_target, model_to_train)

        # TODO: do the backward pass and update the gradients
        loss_local.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss_local.item()


if __name__ == "__main__":
    print("Initiated learning process.")
    agent = Agent(size_of_gamefield=8, animate=False)
    minimal_memory_size = 100
    model_path = None
    plt.style.use('seaborn-whitegrid')

    learning_rate = 0.001
    counter_of_trainings = 0

    hl, = plt.plot([], [])

    trained_model = model.Qnet()

    # torch.autograd.set_detect_anomaly(True)

    if model_path is not None:
        trained_model.load_state_dict(torch.load(model_path, map_location='cpu'))

    optimizer = torch.optim.Adam(trained_model.parameters(), lr=learning_rate)

    target_model = copy.deepcopy(trained_model)

    episode = 0
    while True:
        episode += 1
        # TODO gain more transitions
        game_over = False
        steps = 0
        while not game_over:
            one_transition = agent.get_one_transition(trained_model, expolation_probability=0.1)

            agent.memory_stack.push(one_transition)

            if one_transition[3][0] == DONE:
                game_over = True

            if agent.memory_stack.number_of_items < minimal_memory_size:
                continue

            if game_over:
                loss = agent.train(trained_model, target_model, minimal_memory_size)
                counter_of_trainings += 1
                agent.epochas.append(counter_of_trainings)
                agent.losses.append(loss)
                agent.scores.append(len(agent.game.current_snake.rest_of_body_positions))

        # update target network (copy trained model)
        if episode % 100 == 0:
            print("Episode:", episode,"  Average score:", sum(agent.scores) / 100.0)
            agent.scores = []
            target_model = copy.deepcopy(trained_model)


