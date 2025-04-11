# COMP 5801W - Final Project 
# Carleton University

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy as np
import torch
import time
from queens import QueenWorld

class DuelingDQN(torch.nn.Module):
    def __init__(self, input_channels, grid_size, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, grid_size, grid_size)
            conv_out = self.conv(dummy)
            self.conv_output_size = conv_out.size(1)
        
        self.value_stream = torch.nn.Sequential(
            torch.nn.Linear(self.conv_output_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        
        self.advantage_stream = torch.nn.Sequential(
            torch.nn.Linear(self.conv_output_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions)
        )
    
    def forward(self, x):
        features = self.conv(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

def preprocess_observation(obs, grid_size, num_regions, device):
    queens = obs['queens'].astype(np.float32) / 2.0
    regions = obs['regions'].astype(np.float32)
    if num_regions > 1:
        regions /= (num_regions - 1)
    state = np.stack([queens, regions], axis=0)
    return torch.FloatTensor(state).unsqueeze(0).to(device)

def run_trained_agent():
    grid = np.array(
        [[0, 0, 0, 0, 1], 
         [0, 0, 0, 1, 1], 
         [0, 2, 3, 1, 1],
         [4, 2, 3, 3, 1],
         [4, 2, 2, 3, 3]]
    )
    env = QueenWorld(grid)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    grid_size = grid.shape[0]
    num_actions = env.action_space.n
    num_regions = env.num_regions
    
    model = DuelingDQN(2, grid_size, num_actions).to(device)
    model.load_state_dict(torch.load('dueling_dqn.pth', map_location=device))
    model.eval()
    
    obs = env.reset()
    state = preprocess_observation(obs, grid_size, num_regions, device)
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        with torch.no_grad():
            q_values = model(state)
            action = q_values.argmax().item()
        
        next_obs, reward, done, _ = env.step(action)
        next_state = preprocess_observation(next_obs, grid_size, num_regions, device)
        state = next_state
        total_reward += reward
        print(f"Action: {action}, Reward: {reward}, Total Reward: {total_reward}")
        time.sleep(1)
    
    env.close()
    print(f"Total Reward: {total_reward}")

if __name__ == '__main__':
    run_trained_agent()