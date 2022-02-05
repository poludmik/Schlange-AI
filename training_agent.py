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


def plot_info(epochas, losses, scores, epsilons):

    plt.plot(epochas, scores, label='Score', linewidth=0.5)
    epsilons = [item*10 for item in epsilons]
    plt.plot(epochas, epsilons, label='Exploration rate (10x)')
    # trend
    df = pd.DataFrame(np.transpose(np.array([scores])))
    plt.plot(df.expanding(min_periods=4).mean(), 'b', label='Score moving average of 4')
    plt.legend()
    plt.show()


class Agent:

    def __init__(self, size_of_gamefield=10, animate=False):
        self.size_of_gamefield = size_of_gamefield
        self.game = game_class.SnakeGame(size_of_gamefield, dark=True, window_size=400, animate=animate)
        self.memory_stack = experience_stack.Memory()
        self.losses = []
        self.epochas = []
        self.scores = []
        self.exploration_rates = []

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

    def check_progress(self, model_to_check):
        # play with epsilon=0 and best weights
        while True:
            self.get_one_transition(Qnet_model=model_to_check, expolation_probability=0.0)

if __name__ == "__main__":
    print("Initiated learning process.")

    trained_model = model.Qnet()
    model_path = "best_weights_for_size_10.pth"
    model_path = "best_weights_for_size_8.pth"
    if model_path is not None:
        trained_model.load_state_dict(torch.load(model_path))
        trained_model.eval()

    gamefield_size = 8
    check_progress = True
    if check_progress:
        agent = Agent(size_of_gamefield=gamefield_size, animate=True)
        agent.check_progress(trained_model)
        exit()
    else:
        agent = Agent(size_of_gamefield=gamefield_size, animate=False)


    minimal_memory_size = 100
    plt.style.use('seaborn-whitegrid')
    learning_rate = 0.001
    exploration_epsilon_first = 0.7 # 0.7
    counter_of_trainings = 0
    maximal_score = 0


    optimizer = torch.optim.Adam(trained_model.parameters(), lr=learning_rate)
    target_model = copy.deepcopy(trained_model)
    episode = 0

    while True:
        episode += 1
        game_over = False
        steps = 0
        exploration_epsilon = max(0.01, min(0.7, exploration_epsilon_first / episode * 100)) # works great linearly

        while not game_over:
            one_transition = agent.get_one_transition(trained_model, expolation_probability=exploration_epsilon)
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
                score = len(agent.game.current_snake.rest_of_body_positions)
                agent.scores.append(score)
                agent.exploration_rates.append(exploration_epsilon)
                if score > maximal_score:
                    maximal_score = score
                    torch.save(trained_model.state_dict(), "best_weights_for_size_"+ str(agent.game.width) + ".pth")

        # update target network (copy trained model)
        if episode % 100 == 0:
            print("Episode:", "{:5d}".format(episode),
                  "  Average score:", "{:.2f}".format(sum(agent.scores[-100:]) / 100.0),
                  "  Eps:", "{:.2f}".format(exploration_epsilon))
            plot_info(agent.epochas, agent.losses, agent.scores, agent.exploration_rates)
            target_model = copy.deepcopy(trained_model)


