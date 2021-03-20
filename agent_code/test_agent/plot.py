import math

import matplotlib.pyplot as plt
import numpy as np

def init_plot_data(self):
    self.reward_sum = 0
    self.loss_sum = 0
    self.average_scores = np.array([])
    self.plot_data = {'rewards': [], 'losses': [], 'scores':[]}


def update_plot_data(self, batch_size):
    if self.score > self.high_score:
        self.high_score = self.score
    self.plot_data['rewards'] += [self.reward_sum]
    # loss of evaluation of batch 100
    mean_loss = self.loss_sum / (self.step + batch_size)
    self.plot_data['losses'] += [mean_loss.detach().numpy()]
    self.plot_data['scores'] += [self.score]
    self.reward_sum = 0
    self.loss_sum = 0


def plot(self):
    n_games = range(len(self.plot_data['rewards']))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10))
    ax1.plot(n_games, self.plot_data['rewards'])
    ax1.set_title('Sum of Rewards')
    ax1.set_ylabel('rewards')
    ax2.plot(n_games, self.plot_data['scores'])
    ax2.set_title('Final Score')
    ax2.set_ylabel('score')
    ax3.plot(n_games, self.plot_data['losses'])
    ax3.set_title('Mean Loss at End of Game')
    ax3.set_xlabel('# rounds')
    ax3.set_ylabel('mean MSE loss')
    ax3.set_yscale('log')
    ax4.set_title('average score')
    ax4.set_xlabel('# rounds')
    ax4.set_ylabel('score')
    self.average_scores = np.append(self.average_scores, np.array(self.plot_data['scores']).mean())
    ax4.plot(np.arange(10, n_games.stop + 10, 10), self.average_scores)
    plt.savefig('test_plot.pdf', format='pdf')
    plt.close()
