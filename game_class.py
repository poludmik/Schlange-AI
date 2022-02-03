import pygame
import math
import random
from tqdm import tqdm
import snake

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
NEG_REWARD = -10
POS_REWARD = 10

class SnakeGame:

    def __init__(self, size_of_gamefield=10, dark=True, window_size=400):
        self.window_size = window_size
        self.margin = 1
        self.width = size_of_gamefield
        self.height = size_of_gamefield
        self.cell_width = self.window_size// size_of_gamefield - self.margin
        self.cell_height = self.window_size// size_of_gamefield - self.margin
        self.mode_is_dark = dark
        self.game_running = False
        self.screen = None
        self.clock = None
        self.grid = []
        self.food_position = None
        self.current_snake = None
        if self.mode_is_dark:
            self.tile_color = DARK_BLUE
            self.margin_color = WHITE
        else:
            self.tile_color = WHITE
            self.margin_color = DARK_BLUE

        self.directions = {"left":[-1, 0], "right":[1, 0], "up":[0, -1], "down":[0, 1]}

        self.screen = pygame.display.set_mode([self.window_size + self.margin, self.window_size + self.margin + 30])
        pygame.display.set_caption("Schlange")
        programIcon = pygame.image.load('moray_huge.jpg')
        pygame.display.set_icon(programIcon)
        self.clock = pygame.time.Clock()

        pygame.font.init()
        self.myfont = pygame.font.SysFont('hpsimplifiedjpanregular', 20)

    def play_games(self):
        self.game_running = True
        self.start_new_game()
        quitting = False

        print("Control keys: W,A,D.")

        while not quitting:
            if not self.game_running:
                self.start_new_game()
                self.game_running = True

            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    quitting = True
                    break

                # checking if keydown event happened or not
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.current_snake.make_step_by_given_action(LEFT)
                    if event.key == pygame.K_w:
                        self.current_snake.make_step_by_given_action(STRAIGHT)
                    if event.key == pygame.K_d:
                        self.current_snake.make_step_by_given_action(RIGHT)

        game.end_screen()

    def play_given_action_for_learning(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # If user clicked close button
                quitting = True
                game.end_screen()
                print("Closing window.")

        reward = self.current_snake.make_step_by_given_action(action)

        return reward

    def ensure_game_is_running(self):
        pygame.event.pump()
        if not self.game_running:
            self.start_new_game()
            self.game_running = True

    def get_current_state(self):
        """
        state[11] = danger: [ straight, left, right,
                 direction:   left, right, up, down
                      food:   left, right, up, down]
        """
        state = [0] * 11

        # If there are dangers on sratight, left or right tile
        for idx, action in enumerate([STRAIGHT, LEFT, RIGHT]):
            new_head_direction = self.current_snake.rotate_head_direction(self.current_snake.head_direction, action)
            next_tile = (self.current_snake.head_position[0] + new_head_direction[1],
                         self.current_snake.head_position[1] + new_head_direction[0])

            if (next_tile[0] < 0 or next_tile[0] >= game.width or next_tile[1] < 0 or next_tile[1] >= game.height) or \
                    (next_tile[0], next_tile[1]) in self.current_snake.rest_of_body_positions:
                # next tile in a wall or in the snake body
                state[idx] = 1

        # Direction of a head
        for idx, direction in enumerate(self.directions):
            if self.current_snake.head_direction == self.directions[direction]:
                state[idx + 3] = 1
                break

        # Where is food
        if self.food_position[1] < self.current_snake.head_position[1]:
            state[7] = 1 # food is to the left to the head
        if self.food_position[1] > self.current_snake.head_position[1]:
            state[8] = 1 # food is to the right to the head
        if self.food_position[0] < self.current_snake.head_position[0]:
            state[9] = 1  # food is up to the head
        if self.food_position[0] > self.current_snake.head_position[0]:
            state[10] = 1 # food is down to the head

        return state

    def start_new_game(self):
        self.screen.fill(self.margin_color)
        for row in range(self.width):
            for column in range(self.height):

                pygame.draw.rect(self.screen, self.tile_color,
                                 [(self.margin + self.cell_width) * column + self.margin,
                                  (self.margin + self.cell_height) * row + self.margin,
                                   self.cell_width, self.cell_height]
                                 )
                self.clock.tick(10000)
                pygame.display.flip()

        self.food_position = (random.randint(1, self.width - 2), random.randint(1, self.height - 2))
        self.set_color_to_one_cell(self.food_position[0], self.food_position[1], GREEN)

        if self.current_snake:
            del self.current_snake

        self.current_snake = snake.Snake(self)
        self.current_snake.draw_snake()
        textsurface = self.myfont.render(str(len(self.current_snake.rest_of_body_positions)), False, self.tile_color)
        self.screen.blit(textsurface, (10, self.window_size))
        pygame.display.flip()

    def set_color_to_one_cell(self, r, c, color):
        pygame.draw.rect(self.screen, color,
                         [(self.margin + self.cell_width) * c + self.margin,
                          (self.margin + self.cell_height) * r + self.margin,
                          self.cell_width, self.cell_height]
                         )
        pygame.display.flip()

    def draw_direction(self, r, c, direction):
        # pygame.draw.polygon(self.screen, BLACK, [[100, 100], [0, 200], [200, 200]], 5)
        start_x = (self.margin + self.cell_width) * c + self.margin
        start_y = (self.margin + self.cell_height) * r + self.margin

        if direction == [1, 0]: # right
            pygame.draw.polygon(self.screen, BLACK,
                                [[start_x + self.cell_width - 2*self.margin-1, start_y + 2*self.margin-1],
                                    [start_x + self.cell_width - 2*self.margin-1, start_y + self.cell_height - 2*self.margin-1],
                                    [start_x + self.cell_width // 2, start_y + self.cell_height // 2]], width=0)

        elif direction == [-1, 0]: # left
            pygame.draw.polygon(self.screen, BLACK,
                                [[start_x + 2 * self.margin - 1, start_y + 2 * self.margin - 1],
                                 [start_x + 2 * self.margin - 1, start_y + self.cell_height - 2 * self.margin - 1],
                                 [start_x + self.cell_width // 2, start_y + self.cell_height // 2]], width=0)

        elif direction == [0, 1]:  # down
            pygame.draw.polygon(self.screen, BLACK,
                                [[start_x + 2 * self.margin - 1, start_y + self.cell_height - 2 * self.margin - 1],
                                 [start_x + self.cell_width - 2 * self.margin - 1, start_y + self.cell_height - 2 * self.margin - 1],
                                 [start_x + self.cell_width // 2, start_y + self.cell_height // 2]], width=0)

        elif direction == [0, -1]:  # up
            pygame.draw.polygon(self.screen, BLACK,
                                [[start_x + 2 * self.margin - 1, start_y + 2 * self.margin - 1],
                                 [start_x + self.cell_width - 2 * self.margin - 1, start_y + 2 * self.margin - 1],
                                 [start_x + self.cell_width // 2, start_y + self.cell_height // 2]], width=0)

    def end_screen(self):
        if self.game_running:
            pygame.quit()
            print("Stopped the game, closed the window.")
        else:
            print("No game was running to stop it.")



if __name__ == "__main__":
    print("Starting gaming")

    game = SnakeGame(8, dark=True, window_size=400)

    play_on_keyboard = False

    if play_on_keyboard:
        game.play_games()
    else:
        # for learning, i.e.
        for _ in tqdm(range(30), position=0, leave=True):
            game.ensure_game_is_running()
            # get state
            # decide
            game.play_given_action_for_learning(action=random.choice([LEFT, RIGHT, STRAIGHT]))
            # get reward
            # store episode to memory..?:)
            pygame.time.delay(100)


