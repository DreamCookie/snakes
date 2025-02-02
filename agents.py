import random
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_new_direction, is_danger

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, transition):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class BetterAgent:
    def __init__(self, input_size=6, output_size=3, lr=1e-3, gamma=0.9,
                 epsilon_start=0.1, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_capacity=10000, batch_size=64):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_capacity)
        # Более "умная" архитектура
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state):
        # Epsilon-greedy стратегия
        if random.random() < self.epsilon:
            return random.randrange(3)
        else:
            with torch.no_grad():
                q_values = self.model(state)
            return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
    
    def train_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        batch_state = torch.stack(batch_state)
        batch_action = torch.tensor(batch_action, dtype=torch.long)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float)
        batch_next_state = torch.stack(batch_next_state)
        batch_done = torch.tensor(batch_done, dtype=torch.float)
        
        q_values = self.model(batch_state)
        q_value = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.model(batch_next_state)
            next_q_value = next_q_values.max(1)[0]
            target = batch_reward + self.gamma * next_q_value * (1 - batch_done)
        loss = self.loss_fn(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Декремент epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class WorseAgent:
    def __init__(self, input_size=6, output_size=3, lr=1e-3, gamma=0.9,
                 epsilon_start=0.3, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_capacity=10000, batch_size=64):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_capacity)
        # Архитектура "хуже" – другая конфигурация слоёв и активация Tanh
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, output_size)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(3)
        else:
            with torch.no_grad():
                q_values = self.model(state)
            return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
    
    def train_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        batch_state = torch.stack(batch_state)
        batch_action = torch.tensor(batch_action, dtype=torch.long)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float)
        batch_next_state = torch.stack(batch_next_state)
        batch_done = torch.tensor(batch_done, dtype=torch.float)
        
        q_values = self.model(batch_state)
        q_value = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.model(batch_next_state)
            next_q_value = next_q_values.max(1)[0]
            target = batch_reward + self.gamma * next_q_value * (1 - batch_done)
        loss = self.loss_fn(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

