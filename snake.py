import pygame
import math
import random
from tqdm import tqdm

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (102, 204, 0)
RED = (255, 0, 0)
PALE_BLUE = (102, 178, 255)
STRONGER_BLUE = (102, 102, 255)
DARK_BLUE = (0, 25, 51)
LEFT = [0, 1, 0]
RIGHT = [0, 0, 1]
STRAIGHT = [1, 0, 0]
NEG_REWARD = -10.0
POS_REWARD = 10.0


class Snake:
    def __init__(self, game):
        self.game = game
        self.head_position = self.generate_random_starting_point()
        self.head_direction = random.choice([[1, 0], [0, -1], [-1, 0], [0, 1]])
        self.rest_of_body_positions=[]
        self.length = 1

    def __del__(self):
        #print("Deleted snake")
        pass

    def generate_random_starting_point(self):
        startin_point = (random.randint(1, self.game.width - 2), random.randint(1, self.game.height - 2))
        while startin_point == self.game.food_position:
            startin_point = (random.randint(1, self.game.width - 2), random.randint(1, self.game.height - 2))
        return startin_point

    def make_step_by_given_action(self, action): # action is in form of [0-1, 0-1, 0-1]
        reward = 0.0

        new_head_direction = self.rotate_head_direction(self.head_direction, action)

        next_tile = (self.head_position[0] + new_head_direction[1], self.head_position[1] + new_head_direction[0])

        if next_tile[0] < 0 or next_tile[0] >= self.game.width or next_tile[1] < 0 or next_tile[1] >= self.game.height:
            #print("Wall collision")
            reward = NEG_REWARD
            if self.game.animate:
                self.game.set_color_to_one_cell(self.head_position[0], self.head_position[1], RED)
                self.game.clock.tick(50)
                pygame.display.flip()
            self.game.game_running = False
            return reward

        elif (next_tile[0], next_tile[1]) in self.rest_of_body_positions:
            #print("Body collision")
            reward = NEG_REWARD
            if self.game.animate:
                self.game.set_color_to_one_cell(self.head_position[0], self.head_position[1], RED)
                self.game.clock.tick(50)
                pygame.display.flip()
            self.game.game_running = False
            return reward

        elif next_tile == self.game.food_position:
            # found food
            reward = POS_REWARD
            if self.game.animate:
                self.game.set_color_to_one_cell(self.game.food_position[0], self.game.food_position[1], self.game.tile_color)

            # random food coordinates
            self.game.food_position = (random.randint(0, self.game.width-1), random.randint(0, self.game.height-1))
            while self.game.food_position in self.rest_of_body_positions\
                    or self.game.food_position == self.head_position\
                    or self.game.food_position == next_tile:
                self.game.food_position = (random.randint(0, self.game.width-1), random.randint(0, self.game.height-1))

            if self.game.animate:
                self.game.set_color_to_one_cell(self.game.food_position[0], self.game.food_position[1], GREEN) # draw new food

        # move snake
        if reward != POS_REWARD and len(self.rest_of_body_positions): # if not growing this step, delete tail
            if self.game.animate:
                self.game.set_color_to_one_cell(self.rest_of_body_positions[0][0], self.rest_of_body_positions[0][1],
                                           self.game.tile_color)
            self.rest_of_body_positions.pop(0)
            self.rest_of_body_positions.append(self.head_position)
        if reward != POS_REWARD and len(self.rest_of_body_positions) == 0 and self.game.animate:
            self.game.set_color_to_one_cell(self.head_position[0], self.head_position[1],
                                       self.game.tile_color)
        if reward == POS_REWARD:
            self.rest_of_body_positions.append(self.head_position)

        self.head_position = next_tile
        self.head_direction = new_head_direction
        if self.game.animate:
            self.draw_snake()

        return reward

    def draw_snake(self):
        self.game.set_color_to_one_cell(self.head_position[0], self.head_position[1], STRONGER_BLUE)

        # Draw snake fragments
        for body_fragment in self.rest_of_body_positions:
            self.game.set_color_to_one_cell(body_fragment[0], body_fragment[1], PALE_BLUE)
        self.game.draw_direction(self.head_position[0], self.head_position[1], self.head_direction)

        # Show snake score
        textsurface = self.game.myfont.render(str(len(self.game.current_snake.rest_of_body_positions)), False, self.game.tile_color)
        pygame.draw.rect(self.game.screen, self.game.margin_color, [0, self.game.window_size, self.game.window_size, 30])
        self.game.screen.blit(textsurface, (10, self.game.window_size))

        pygame.display.flip()

        self.game.clock.tick(10)

    def rotate_head_direction(self, vector, where):
        if where == LEFT:    # -pi/2
            return [vector[1], -vector[0]]
        elif where == RIGHT: # pi/2
            return [-vector[1], vector[0]]
        else:                # no change
            return vector
