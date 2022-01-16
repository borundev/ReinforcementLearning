import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import wandb

env = gym.make("MountainCar-v0")
input_shape = 2  # == env.observation_space.shape
n_outputs = 3

xs=np.linspace(-1.2,0.5,100)
vs=np.linspace(-0.07,0.07,100)

X, V = np.meshgrid(xs, vs)

phase_space_points=torch.stack([torch.Tensor(X.flatten()),torch.Tensor(V.flatten())],1)

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, n_outputs)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        return self.model(x)


class Tyler(nn.Module):
    def __init__(self, n, l, xmin=-1.21, xmax=.51, ymin=-.071, ymax=.071, eta=1.2):
        super().__init__()
        dxt = eta * (xmax - xmin)
        dyt = eta * (ymax - ymin)

        self.dXt = np.array([dxt, dyt])

        xmin0 = xmax - dxt
        ymin0 = ymax - dyt

        self.l = l
        self.n = n

        x0 = np.random.uniform(xmin0, xmin, self.l - 2)
        y0 = np.random.uniform(ymin0, ymin, self.l - 2)
        x0 = np.concatenate([[xmin0], x0, [xmin]])
        y0 = np.concatenate([[ymin0], y0, [ymin]])

        self.tile0 = np.stack([x0, y0], axis=1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        x = (((x[:, None, :] - self.tile0[None, ...]) / self.dXt * self.n).type(torch.int) * torch.Tensor(
            [1, self.n])).sum(
            axis=2)
        x = torch.as_tensor(x).type(torch.LongTensor)
        x = torch.nn.functional.one_hot(x, num_classes=self.n ** 2).reshape(-1, self.l * self.n ** 2)
        return x.type(torch.float32)


class ModelLinear(nn.Module):

    def __init__(self, n=8, l=8):
        super().__init__()
        self.model = nn.Sequential(
            Tyler(n, l),
            nn.Linear(n * n * l, n_outputs),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        return self.model(x)


model = Model()
target_model = Model()

def init_weights(m):
    if isinstance(m, nn.Linear):
        #torch.nn.init.xavier_uniform_(m.weight,.01)
        m.bias.data.fill_(0.0)

target_model=model.apply(init_weights)
model=model.apply(init_weights)

def epsilon_greedy_policy(state, epsilon=0):
    """
    Takes a random action with probability epsilon and a greedy one with probability 1-epsilon
    """
    if torch.rand(1) < epsilon:
        return torch.randint(3, (1,))[0]
    else:
        with torch.no_grad():
            Q_values = model(state[np.newaxis])
        return torch.argmax(Q_values[0])


from collections import deque

replay_buffer = deque(maxlen=10000)


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


def training_step(batch_size,double=False):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    if not double:
        with torch.no_grad():
            next_Q_values = target_model(next_states)
        max_next_Q_values = torch.max(next_Q_values, axis=1).values
    else:
        with torch.no_grad():
            next_Q_values = model(next_states)
        best_next_actions = np.argmax(next_Q_values, axis=1)
        next_mask = torch.nn.functional.one_hot(best_next_actions, n_outputs)
        with torch.no_grad():
            max_next_Q_values = (target_model(next_states) * next_mask).sum(axis=1)

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


batch_size = 16
discount_factor = 0.99
episodes = 5000
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()



wandb.init(project="RL Cartpole", entity="borundev")

obs = env.reset()
for step in range(1000):
    epsilon = 1
    obs, reward, done, info = play_one_step(env, obs, epsilon)
    if done:
        obs = env.reset()


global_step=0
max_episode_reward = -200
for episode in range(episodes):
    obs = env.reset()
    done=False
    step=0
    episode_reward=0
    x0=v0=float('inf')
    x1=v1=-float('inf')

    while not done:
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        x, v = obs
        x0 = min(x0, x)
        x1 = max(x1, x)
        v0 = min(v0, v)
        v1 = max(v1, v)

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        step+=1
        episode_reward += reward
        global_step+=1

        loss = training_step(batch_size)
        #wandb.log({"loss": loss})
        #wandb.log({"x": x})
        #wandb.log({"v": v})
    if episode_reward > max_episode_reward:
        torch.save(model, 'best_model_weights.pth')
        max_episode_reward = episode_reward

    wandb.log({"episode": episode + 1})
    wandb.log({"steps": step+1})
    wandb.log({"episode_reward":episode_reward})
    wandb.log({"min x": x0})
    wandb.log({"max x": x1})
    wandb.log({"min v": v0})
    wandb.log({"max v": v1})
    print('Episode {} Rewards {}'.format(episode+1,episode_reward),end='\r',flush=True)

    if episode % 50 == 0:
        with torch.no_grad():
            z = target_model(phase_space_points)
        z = z.reshape(100, 100, 3)
        Z = z.max(dim=2)[0].numpy()

        fig = plt.figure(figsize=(15,15))
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, V, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        ax.set_zlabel('V*')
        wandb.log({'value function': wandb.Image(ax)})
        plt.close()


wandb.config = {
    "learning_rate": lr,
    "discount_factor": discount_factor,
    "episodes": episodes,
    "batch_size": batch_size,
}

torch.save(model, 'model_weights.pth')
torch.save(target_model, 'target_model_weights.pth')

#fig, ax = plt.subplots(figsize=(15,5))
#ax.plot(total_rewards)
#wandb.log({'total_rewards':wandb.Image(ax)})
