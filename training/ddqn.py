import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from queens import QueenWorld

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DuelingDQN(nn.Module):
    def __init__(self, input_channels, grid_size, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, grid_size, grid_size)
            conv_out = self.conv(dummy)
            self.conv_output_size = conv_out.size(1)
        
        self.value_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    
    def forward(self, x):
        features = self.conv(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = state.squeeze(0).cpu().numpy()
        next_state = next_state.squeeze(0).cpu().numpy()
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env, grid_size, num_regions, device):
        self.env = env
        self.grid_size = grid_size
        self.num_regions = num_regions
        self.device = device
        self.num_actions = env.action_space.n
        
        input_channels = 2
        self.policy_net = DuelingDQN(input_channels, grid_size, self.num_actions).to(device)
        self.target_net = DuelingDQN(input_channels, grid_size, self.num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.replay_buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 1000
        self.steps_done = 0
    
    def preprocess(self, obs):
        queens = obs['queens'].astype(np.float32) / 2.0
        regions = obs['regions'].astype(np.float32)
        if self.num_regions > 1:
            regions /= (self.num_regions - 1)
        state = np.stack([queens, regions], axis=0)
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def learn(self):
        samples = self.replay_buffer.sample(self.batch_size)
        if samples is None:
            return None
        states, actions, rewards, next_states, dones = samples
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

def train_agent(env, device, episodes=1000):
    grid_size = env.grid_size
    num_regions = env.num_regions
    
    agent = DQNAgent(env, grid_size, num_regions, device)
    
    returns = []
    
    for episode in range(episodes):
        obs = env.reset()
        state = agent.preprocess(obs)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = agent.preprocess(next_obs)
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            
            state = next_state
            total_reward += reward
        
        returns.append(total_reward)
        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    torch.save(agent.policy_net.state_dict(), 'dueling_dqn.pth')
    
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Returns')
    plt.savefig('training_returns.png')
    plt.close()

if __name__ == '__main__':
    grid = np.array(
        [[0, 0, 0, 0, 1], 
         [0, 0, 0, 1, 1], 
         [0, 2, 3, 1, 1],
         [4, 2, 3, 3, 1],
         [4, 2, 2, 3, 3]]
    )
    env = QueenWorld(grid)
    train_agent(env, device, episodes=1000)