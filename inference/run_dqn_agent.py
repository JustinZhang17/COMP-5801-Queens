# COMP 5801W - Final Project 
# Carleton University

import os
import numpy as np
import torch
import torch.nn as nn
import pygame
from queens import QueenWorld  # Ensure queen_env.py is in the same directory

# Set device for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use same network architecture as in dqn.py
class DQN(nn.Module):
    def __init__(self, grid_size, num_actions):
        super(DQN, self).__init__()
        # Three convolutional layers with increased channels
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
        # Fully connected layers: first layer to reduce conv features, then output layer
        conv_output_size = 128 * grid_size * grid_size
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def process_observation(obs):
    # Convert the observation dict into a tensor with shape (2, grid_size, grid_size)
    queens = obs['queens']
    regions = obs['regions']
    state = np.stack([queens, regions], axis=0).astype(np.float32)
    return state


def run_agent(model_path, delay=100):
    # Define the grid (must be same as used during training)
    grid = np.array(
                [[0, 0, 0, 0, 1], 
                 [0, 0, 0, 1, 1], 
                 [0, 2, 3, 1, 1],
                 [4, 2, 3, 3, 1],
                 [4, 2, 2, 3, 3]])

    env = QueenWorld(grid)
    grid_size = env.grid_size
    num_actions = env.action_space.n

    # Load the trained model
    policy_net = DQN(grid_size, num_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()
    print("Model loaded. Running agent...")

    obs = env.reset()
    state = process_observation(obs)
    done = False

    while not done:
        # Greedily select the best action (no epsilon exploration during testing)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        action = q_values.argmax().item()

        obs, reward, done, _ = env.step(action)
        state = process_observation(obs)

        env.render()
        # Delay to make rendering observable
        pygame.time.wait(delay)

    env.close()
    print("Episode finished.")

if __name__ == "__main__":
    # Path to the saved model file (adjust if necessary)
    model_path = os.path.join("saved_models", "dqn_model.pth")
    run_agent(model_path)
