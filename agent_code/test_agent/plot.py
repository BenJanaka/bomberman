import matplotlib.pyplot as plt

def update_plot_data(self):
    self.plot_data['rewards'] += [self.reward_sum]
    # loss of evaluation of batch 100
    self.plot_data['losses'] += [self.loss]
    self.plot_data['scores'] += [self.score]

def plot(self, laststate):
    n_games = range(len(self.plot_data['rewards']))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,13))
    ax1.plot(n_games, self.plot_data['rewards'])
    ax1.set_title('Sum of Rewards')
    ax2.plot(n_games, self.plot_data['scores'])
    ax2.set_title('Final Score')
    ax3.plot(n_games, self.plot_data['losses'])
    ax3.set_title('Loss')
    plt.show()
    pass
