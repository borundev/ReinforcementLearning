import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import wandb

env = gym.make("CartPole-v1")
input_shape = [4]  # == env.observation_space.shape
n_outputs = 2


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, n_outputs)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        return self.model(x)


model = Model()


def epsilon_greedy_policy(state, epsilon=0):
    if torch.rand(1) < epsilon:
        return torch.randint(2, (1,))[0]
    else:
        with torch.no_grad():
            Q_values = model(state[np.newaxis])
        return torch.argmax(Q_values[0])


from collections import deque

replay_buffer = deque(maxlen=2000)


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    dones = torch.tensor(dones).type(torch.int)
    rewards = torch.tensor(rewards).type(torch.float32)
    actions = torch.tensor(actions)
    return states, actions, rewards, next_states, dones


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon).numpy()
    next_state, reward, done, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    with torch.no_grad():
        next_Q_values = model(next_states)
    max_next_Q_values = torch.max(next_Q_values, axis=1).values
    target_Q_values = (rewards +
                       (1 - dones) * discount_factor * max_next_Q_values)
    mask = torch.nn.functional.one_hot(actions, n_outputs)
    all_Q_values = model(states)
    Q_values = torch.sum(all_Q_values * mask, 1)
    loss = loss_fn(Q_values, target_Q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


batch_size = 32
discount_factor = 0.95
episodes = 600
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

wandb.init(project="RL Cartpole", entity="borundev")

total_rewards = []
for episode in range(600):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    total_rewards.append(step + 1)
    if episode > 50:
        loss=training_step(batch_size)
        wandb.log({"loss": loss})

wandb.config = {
    "learning_rate": lr,
    "discount_factor": discount_factor,
    "episodes": episodes,
    "batch_size": batch_size,
}


fig, ax = plt.subplots(figsize=(15,5))
ax.plot(total_rewards)
wandb.log({'total_rewards':wandb.Image(ax)})
