from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from torch import nn
from tqdm import tqdm

class CompetingBanditUCB(nn.Module):

    def __init__(self, payoff_matrix, num_bandits, epsilon=.01):
        super().__init__()
        self.payoff_matrix = payoff_matrix
        self.num_actions = len(self.payoff_matrix)
        self.num_bandits = num_bandits
        self.epsilon = epsilon

        self.qs = np.zeros([self.num_bandits, self.num_actions])
        self.steps = np.zeros([self.num_bandits, self.num_actions])
        self.total_steps = np.zeros([self.num_bandits, 1])

    def play_one(self):
        if (self.steps > 0).all():
            return np.argmax(self.qs + np.sqrt(np.log(self.total_steps) / (self.steps + 1e-5)), -1)
        else:
            return np.random.randint(0, self.num_actions, self.num_bandits)

    def get_rewards(self, actions):
        # logic of inverting shuffle based on https://stackoverflow.com/questions/26577517/inverse-of-random-shuffle

        # shuffle the actions of agents and pair them to get rewards

        # Here we create an array of shuffled indices
        shuf_order = np.arange(self.num_bandits)
        np.random.shuffle(shuf_order)
        shuffled_actions = actions[shuf_order]  # Shuffle the original data

        # Create an inverse of the shuffled index array (to reverse the shuffling operation, or to "unshuffle")
        unshuf_order = np.zeros_like(shuf_order)
        unshuf_order[shuf_order] = np.arange(self.num_bandits)

        # unshuffled_actions = shuffled_actions[unshuf_order]  # Unshuffle the shuffled data
        # assert np.all(np.equal(unshuffled_actions, actions))

        shuffled_mask = np.ones(self.num_bandits, dtype=np.bool)
        if self.num_bandits % 2 == 1:
            shuffled_mask[-1] = False
            shuffled_actions = shuffled_actions[:-1]

        paired_actions = shuffled_actions.reshape(self.num_bandits // 2, 2)
        rows = paired_actions[:, 0]
        cols = paired_actions[:, 1]

        shuffled_rewards = self.payoff_matrix[rows, cols].reshape(self.num_bandits >> 1 << 1, )
        # assert all([k in payoff_matrix.sum(-1).flatten() for k in shuffled_rewards.reshape(-1,2).sum(1) ])

        mask = shuffled_mask[unshuf_order]
        # return unshuffled rewards
        if self.num_bandits % 2 == 1:
            shuffled_rewards = np.concatenate([shuffled_rewards, [201]])
        rewards = shuffled_rewards[unshuf_order]

        return rewards, mask

    def take_rewards(self, r, actions, mask):
        idx = np.arange(self.num_bandits)[mask]
        actions = actions[mask]
        r = r[mask]
        self.steps[idx, actions] += 1
        self.total_steps[idx] += 1
        self.qs[idx, actions] += 1.0 / 50 * (r - self.qs[idx, actions])


if __name__ == '__main__':
    c, a, d, b = 0, -1, -2, -3
    payoff_matrix = {"Prisoner's Dillemma": np.array([[(a, a), (b, c)], [(c, b), (d, d)]]),
                   'Rock Paper Scissor' : np.array(
        [
            [(0, 0), (-1, 1), (1, -1)],
            [(1, -1), (0, 0), (-1, 1)],
            [(-1, 1), (1, -1), (0, 0)]
        ],
        dtype=np.float64
    )}

    game = "Prisoner's Dillemma"
    cb = CompetingBanditUCB(payoff_matrix[game], 2)
    wandb.init(project="Competing Bandits", entity="borundev", name = game)

    actions = []

    for s in tqdm(range(100000)):
        try:
            a = cb.play_one()
            r,m = cb.get_rewards(a)
            cb.take_rewards(r, a,m)
            actions.append(a)

        except KeyboardInterrupt:
            break

    _,ax=plt.subplots(1,1)
    ax.bar(range(cb.num_actions), cb.qs.mean(0), yerr=cb.qs.mean(0) / np.sqrt(cb.num_bandits))
    wandb.log({'Q function': wandb.Image(ax)})
    plt.close()

    actions = np.stack(actions)
    _, ax = plt.subplots(1, 1)
    a,b=list(zip(*Counter(actions[5000:].flatten()).items()))
    ax.bar(a,b)
    wandb.log({'Actions': wandb.Image(ax)})
    plt.close()





