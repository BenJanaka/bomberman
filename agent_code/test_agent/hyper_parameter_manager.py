from itertools import product


class HyperParameterManager:
    def __init__(self,
                 transition_history_size=100000,
                 batch_size=100,
                 exploration_prob=0.25,
                 learning_rate=0.001,
                 gamma=0.9,
                 tau=None,
                 mode=None):
        if mode == 'HP-TEST':
            self.parameters = dict(
                transition_history_size=[10000, 100000],
                batch_size=[100, 256],
                learning_rate=[0.001, 0.0001],
                gamma=[0.8, 0.9, 0.99],
                tau=[2, 4]
            )
            param_values = [v for v in self.parameters.values()]
            self.param_product = list(product(*param_values))

        self.transition_history_size = transition_history_size
        self.batch_size = batch_size
        self.exploration_prob = exploration_prob
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
