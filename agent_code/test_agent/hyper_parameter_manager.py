from itertools import product


class HyperParameterManager:
    def __init__(self,
                 transition_history_size=100000,
                 batch_size=100,
                 epsilon_decay=0.0001,
                 learning_rate=0.001,
                 exploration_prob=0.9,
                 gamma=0.9,
                 tau=None,
                 mode=None):
        if mode == 'HP-TEST':
            self.parameters = dict(
                transition_history_size=[10000],
                batch_size=[100, 256],
                learning_rate=[0.001, 0.0001],
                gamma=[0.8, 0.9, 0.99],
                tau=[2]
            )
            # self.parameters = dict(
            #     transition_history_size=[10000],
            #     batch_size=[100],
            #     learning_rate=[0.0001],
            #     gamma=[0.99],
            #     tau=[1.5, 2, 2.5, 3.5, 4]
            # )
            # self.parameters = dict(
            #     transition_history_size=[10000],
            #     batch_size=[100],
            #     learning_rate=[0.0001],
            #     gamma=[0.99],
            #     epsilon_decay=[0.0001, 0.0005, 0.001, 0.005, 0.01]
            # )
            param_values = [v for v in self.parameters.values()]
            self.param_product = list(product(*param_values))

        self.transition_history_size = transition_history_size
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.exploration_prob = exploration_prob

    def next_params(self, run_id):
        self.transition_history_size = self.param_product[run_id][0]
        self.batch_size = self.param_product[run_id][1]
        self.learning_rate = self.param_product[run_id][2]
        self.gamma = self.param_product[run_id][3]
        self.tau = self.param_product[run_id][4]
