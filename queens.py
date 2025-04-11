# COMP 5801W - Final Project 
# Carleton University

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

class QueenWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid):
        super(QueenWorld, self).__init__()

        if grid is None:
            pass

        if grid.shape[0] != grid.shape[1]:
            pass

        self.grid_size = grid.shape[0] 

        self.regions = grid 

        unique_regions = np.unique(self.regions)
        self.num_regions = len(unique_regions)

        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
    
        # Observation space
        self.observation_space = spaces.Dict({
            'queens': spaces.Box(low=0, high=2, shape=(self.grid_size, self.grid_size), dtype=int),
            'regions': spaces.Box(low=0, high=self.num_regions-1, shape=(self.grid_size, self.grid_size), dtype=int)
        })

        # Pygame setup
        self.screen = None
        self.cell_size = 50
        self.padding = 40
        self.colors = None

        # Game state
        self.queens = np.zeros((self.grid_size, self.grid_size))
        self.attempts = 0
        self.queens_placed = 0
        self.crosses_placed = 0

    def reset(self):
        self.queens_placed = 0
        self.crosses_placed = 0
        self.queens = np.zeros((self.grid_size, self.grid_size))
        return self._get_obs()
    
    def _get_obs(self):
        return {
            'queens': self.queens.copy(),
            'regions': self.regions.copy()
        }

    
    def step(self, action):
        self.attempts += 1
        x = action // self.grid_size
        y = action % self.grid_size
        reward = 0

        new_state = (self.queens[x, y] + 1) % 3
        self.queens[x, y] = new_state

        if new_state == 1:
            self.crosses_placed += 1
        elif new_state == 2:
            self.crosses_placed -= 1
            self.queens_placed += 1
        elif new_state == 0:
            self.queens_placed -= 1

        # Queen rewards 
        QUEEN_ADD = 1
        QUEEN_REMOVE = -10
        QUEEN_INVALID = -10
        QUEEN_VALID = 1

        # Cross rewards 
        CROSS_ADD = 0.5
        CROSS_REMOVE = -0.5
        CROSS_INVALID = -1

        # remove reward if a queen is placed into a region with another queen
        if self.queens[x, y] == 2: 
            region = self.regions[y, x]
            region_queens = np.sum(self.queens[np.transpose(self.regions) == region] == 2)
            if region_queens > 1:
                reward += QUEEN_INVALID 
            
        # remove reward if a queen is placed in a column with another queen
        if self.queens[x, y] == 2:
            col_queens = np.sum(self.queens[x, :] == 2)
            if col_queens > 1:
                reward += QUEEN_INVALID 

        # remove reward if a queen is placed in a row with another queen
        if self.queens[x, y] == 2:
            row_queens = np.sum(self.queens[:, y] == 2)
            if row_queens > 1:
                reward += QUEEN_INVALID 

        # remove reward if a queen is adjecent diagonally to another queen
        if self.queens[x, y] == 2:
            for i in [-1, 1]:
                for j in [-1, 1]:
                    if x+i >= 0 and x+i < self.grid_size and y+j >= 0 and y+j < self.grid_size:
                        if self.queens[x+i, y+j] == 2:
                            reward += QUEEN_INVALID

        # add reward if a queen is placed
        if self.queens[x, y] == 2:
            reward += QUEEN_ADD

        # remove reward if a queen is removed
        if self.queens[x, y] == 0:
            reward += QUEEN_REMOVE 

        # add reward if a region only has 1 queen
        if self.queens[x, y] == 2:
            region = self.regions[y, x]
            region_queens = np.sum(self.queens[np.transpose(self.regions) == region] == 2)
            if region_queens == 1:
                reward += QUEEN_VALID 

        # add reward if a column only has 1 queen
        if self.queens[x, y] == 2:
            col_queens = np.sum(self.queens[x, :] == 2)
            if col_queens == 1:
                reward += QUEEN_VALID 

        # add reward if a row only has 1 queen
        if self.queens[x, y] == 2:
            row_queens = np.sum(self.queens[:, y] == 2)
            if row_queens == 1:
                reward += QUEEN_VALID 


        # add reward if a cross is placed
        if self.queens[x, y] == 1:
            reward += CROSS_ADD 

        # remove reward if a cross is removed
        if self.queens[x, y] == 2:
            reward += CROSS_REMOVE 

        # remove reward if crosses fill an entire region
        if self.queens[x, y] == 1:
            region = self.regions[y, x]
            region_crosses = np.sum(self.queens[np.transpose(self.regions) == region] == 1)
            if region_crosses == np.sum(self.regions == region):
                reward += CROSS_INVALID

        # remove reward if crosses fill an entire column 
        if self.queens[x, y] == 1:
            col_crosses = np.sum(self.queens[x, :] == 1)
            if col_crosses == self.grid_size:
                reward += CROSS_INVALID 

        # remove reward if crosses fill an entire row 
        if self.queens[x, y] == 1:
            row_crosses = np.sum(self.queens[:, y] == 1)
            if row_crosses == self.grid_size:
                reward += CROSS_INVALID


        if self.attempts == self.grid_size * self.grid_size * 2:
            done = True
            self.attempts = 0
        else:
            done = False

        return self._get_obs(), reward, done, {}



    def render(self, mode='human'):

        # Load Queen and Cross images
        if not hasattr(self, 'star_sprite') or not hasattr(self, 'cross_sprite'):
            try:
                # Load and scale sprite (replace 'star.png' with your image path)
                self.star_sprite = pygame.image.load('./assets/queen.png')
                # Scale to 50% of cell size
                sprite_size = int(self.cell_size * 0.5)
                self.star_sprite = pygame.transform.scale(
                    self.star_sprite, 
                    (sprite_size, sprite_size))

                self.cross_sprite = pygame.image.load('./assets/cross.png')
                # Scale to 25% of cell size
                sprite_size = int(self.cell_size * 0.25)
                self.cross_sprite = pygame.transform.scale(
                    self.cross_sprite, 
                    (sprite_size, sprite_size))
            except FileNotFoundError:
                self.star_sprite = None
                self.cross_sprite = None

        s = self.cell_size
        p = self.padding

        if self.screen is None:
            # Calculate total window size with padding
            window_width = self.grid_size * s + 2 * p
            window_height = self.grid_size * s + 2 * p
            
            pygame.init()
            self.screen = pygame.display.set_mode((window_width, window_height))

            pygame.display.set_caption('Queens - Puzzle Game')

        self.colors = {
            0: (179,223,160),
            1: (187,163,226),
            2: (255,201,146),
            3: (150,190,255),
            4: (223,223,223),
            5: (255,123,96),
            6: (230,243,136),
            7: (185,178,158),
            8: (223,160,191),
            9: (163,210,216)
        }

        self.screen.fill((255, 255, 255))

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                color = self.colors[self.regions[x, y]]  
                pygame.draw.rect(
                    self.screen,
                    color,
                    (p + y * s, p + x * s, s, s),
                )

        # Draw grid lines with padding offset
        pygame.draw.line(self.screen, (0,0,0), (p, p), (p + self.grid_size * s, p), 3)
        pygame.draw.line(self.screen, (0,0,0), (p, p), (p, p + self.grid_size * s), 3)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                next = None if x+1 == self.grid_size else self.regions[y,x+1]
                width = 3 if self.regions[y,x] != next else 1
                pygame.draw.line(self.screen, (0,0,0),
                                (p + (x+1) * s, p + (y * s)),
                                (p + (x+1) * s, p + (y+1) * s), width)

                next = None if y+1 == self.grid_size else self.regions[y+1,x]
                width = 3 if self.regions[y,x] != next else 1
                pygame.draw.line(self.screen, (0,0,0),
                                (p + (x * s), p + (y+1) * s),
                                (p + (x+1) * s, p + (y+1) * s), width)

        # Draw queens and crosses
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.queens[j, i] == 1:
                    center = (p + j * s + s // 2, p + i * s + s // 2)
                    if self.cross_sprite:
                        star_rect = self.cross_sprite.get_rect(
                                        center=(center[0], center[1]))
                        self.screen.blit(self.cross_sprite, star_rect)
                    else:
                        pygame.draw.circle(self.screen, (0, 255, 0), center, s // 4)

                if self.queens[j, i] == 2:
                    center = (p + j * s + s // 2, p + i * s + s // 2)
                    if self.star_sprite:
                        star_rect = self.star_sprite.get_rect(
                                        center=(center[0], center[1]))
                        self.screen.blit(self.star_sprite, star_rect)
                    else:
                        pygame.draw.circle(self.screen, (255, 0, 0), center, s // 4)

        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def click(self, pos):
        x, y = pos
        
        # Subtract padding from coordinates
        x -= self.padding
        y -= self.padding
        
        # Convert to grid coordinates
        i = y // self.cell_size
        j = x // self.cell_size

        action = j * self.grid_size + i

        if i < self.grid_size and j < self.grid_size and i >= 0 and j >= 0:
            return action

if __name__ == '__main__':
    grid = np.array(
                [[0, 0, 0, 0, 1], 
                 [0, 0, 0, 1, 1], 
                 [0, 2, 3, 1, 1],
                 [4, 2, 3, 3, 1],
                 [4, 2, 2, 3, 3]])

    # grid = np.array(
    #             [[1,1,0,0,0,0,0],
    #              [1,1,0,0,0,2,0],
    #              [0,0,3,3,3,0,0],
    #              [0,0,3,4,4,0,0],
    #              [0,0,3,4,0,0,0],
    #              [0,5,0,0,0,0,6],
    #              [0,0,0,0,6,6,6]])


    # grid = np.array(
    #             [[8, 8, 1, 1, 2, 3, 3, 3, 3], 
    #              [8, 8, 1, 2, 2, 2, 7, 7, 3],
    #              [8, 8, 5, 5, 2, 5, 5, 7, 3],
    #              [8, 5, 5, 5, 5, 5, 5, 5, 3],
    #              [8, 5, 5, 5, 5, 5, 5, 5, 3],
    #              [8, 8, 5, 5, 5, 5, 5, 3, 3],
    #              [8, 8, 6, 5, 5, 5, 3, 3, 4],
    #              [8, 6, 6, 6, 5, 0, 3, 3, 4],
    #              [8, 8, 8, 6, 0, 0, 4, 4, 4]])

    env = QueenWorld(grid)
    env.reset()
    env.render()
    done = False
    r = 0

    while not done:
        # NOTE: Human Input
        # for event in pygame.event.get():
        #     if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        #         action = env.click(event.pos)
        
        #         # action = np.random.randint(env.action_space.n)
        #         if action is not None:
        #             _, reward, done, _ = env.step(action)
        #             r += reward
        #             print(f"Reward: {reward}, Total Reward: {r}")
        #         env.render()

        action = np.random.randint(env.action_space.n)
        _, reward, done, _ = env.step(action)
        r += reward
        env.render()
        # pygame.time.wait(100)
    print(r)
    env.close()





