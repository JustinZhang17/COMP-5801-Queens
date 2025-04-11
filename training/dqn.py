import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt  # For plotting

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the environment.
# Ensure your environment code (the provided QueenWorld) is saved as queens.py
from queens import QueenWorld  # Adjust the import if your file name is different

# Define the grid you want to use (customize as needed)
grid = np.array(
    [[0, 0, 0, 0, 1],
     [0, 0, 0, 1, 1],
     [0, 2, 3, 1, 1],
     [4, 2, 3, 3, 1],
     [4, 2, 2, 3, 3]]
)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Unzip the batch tuples
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# --- Updated Q-Network with increased capacity ---
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


# --- Helper Functions ---
def process_observation(obs):
    """
    Convert the observation dict into a PyTorch tensor.
    The observation is a dict with 'queens' and 'regions', each a (grid_size, grid_size) array.
    We stack them into a 2-channel tensor and convert to float.
    """
    queens = obs['queens']
    regions = obs['regions']
    state = np.stack([queens, regions], axis=0).astype(np.float32)
    return state


def select_action(state, epsilon, policy_net, num_actions):
    """
    Epsilon-greedy action selection.
    """
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        # Prepare state tensor (add batch dimension)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return q_values.argmax().item()


# --- Hyperparameters (adjusted) ---
num_episodes = 300            # Total number of episodes
batch_size = 128
gamma = 0.99
epsilon_start = 1.0
epsilon_final = 0.05
epsilon_decay = 1000          # Increased epsilon decay constant to maintain exploration longer
learning_rate = 1e-3          # Increased learning rate for faster learning updates
target_update_freq = 10       # How often to update the target network
replay_capacity = 10000

# Create directory to save the model
save_path = "./saved_models"
os.makedirs(save_path, exist_ok=True)
print(f"Model will be saved to {save_path}")

# --- Initialize Environment, Networks, and Optimizer ---
env = QueenWorld(grid)
grid_size = env.grid_size
num_actions = env.action_space.n

policy_net = DQN(grid_size, num_actions).to(device)
target_net = DQN(grid_size, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(replay_capacity)

# For plotting: store returns per episode
episode_returns = []

# --- Training Loop ---
steps_done = 0

for episode in range(num_episodes):
    obs = env.reset()
    state = process_observation(obs)
    done = False
    episode_reward = 0

    while not done:
        # Compute epsilon using a slower decay
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * steps_done / epsilon_decay)
        steps_done += 1

        # Epsilon-greedy action selection
        action = select_action(state, epsilon, policy_net, num_actions)
        next_obs, reward, done, _ = env.step(action)
        next_state = process_observation(next_obs)
        episode_reward += reward

        # Save the transition in replay memory
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        # Optimize the model if enough experiences are available
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            # Convert batch data to tensors
            states = torch.from_numpy(np.array(states)).to(device)
            actions = torch.from_numpy(actions).long().to(device)
            rewards = torch.from_numpy(rewards).float().to(device)
            next_states = torch.from_numpy(np.array(next_states)).to(device)
            dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)

            # Compute Q-values for the current states and actions
            q_values = policy_net(states)
            state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute Q-values for next states from the target network
            with torch.no_grad():
                next_q_values = target_net(next_states)
                next_state_values = next_q_values.max(1)[0]
            expected_state_action_values = rewards + gamma * next_state_values * (1 - dones)

            # Compute loss (Mean Squared Error)
            loss = nn.MSELoss()(state_action_values, expected_state_action_values)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update the target network periodically
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Log the return of the episode for plotting
    episode_returns.append(episode_reward)
    print(f"Episode {episode+1}/{num_episodes}, Return: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")

# Save the trained model
model_file = os.path.join(save_path, "dqn_model.pth")
torch.save(policy_net.state_dict(), model_file)
print(f"Training complete. Model saved to {model_file}")

env.close()

# --- Plotting the Returns ---
plt.figure(figsize=(12, 6))
plt.plot(episode_returns, label="Episode Return")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Episode Returns over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "training_returns.png"))
plt.show()
