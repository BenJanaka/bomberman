from torch.utils.tensorboard import SummaryWriter
import sys

from .hyper_parameter_manager import HyperParameterManager


class TensorBoardManager:
    def __init__(self, hyper_parameter_manager: HyperParameterManager, log_dir: str):
        self.hpm = hyper_parameter_manager
        self.run_id = 0
        self.log_dir = log_dir

        self.summary_writer = self.create_summary_writer()

        self.score = None
        self.total_loss = None
        self.high_score = None
        self.reward_sum = None
        self.loss_sum = None
        self.n_suicides = None
        self.current_epoch = None

        self.reset_metric_values()

    def create_summary_writer(self):
        if self.run_id > len(self.hpm.param_product):
            return
        title = '_'
        for idx, (key, value) in enumerate(self.hpm.parameters.items()):
            title += key + '=' + str(self.hpm.param_product[self.run_id][idx]) + '_'
        sw = SummaryWriter(log_dir=self.log_dir + title)
        self.hpm.next_params(self.run_id)
        return sw

    # increment the run id and reset the values
    def prepare_next_training_instance(self):
        self.set_hparams()

        self.run_id += 1
        self.summary_writer = self.create_summary_writer()
        self.reset_metric_values()


    def reset_metric_values(self):
        self.score = 0
        self.total_loss = 0
        self.high_score = float("-inf")
        self.reward_sum = 0
        self.loss_sum = 0
        self.n_suicides = 0
        self.current_epoch = 1

    def set_hparams(self):
        self.summary_writer.add_hparams(
            {
                "transition_history_size": self.hpm.param_product[self.run_id][0],
                "batch_size": self.hpm.param_product[self.run_id][1],
                "learning_rate": self.hpm.param_product[self.run_id][2],
                "gamma": self.hpm.param_product[self.run_id][3],
                "tau": self.hpm.param_product[self.run_id][4],
            },
            {
                "loss": self.total_loss
            }
        )

    def add_plot_data(self, loss):
        self.summary_writer.add_scalar("Loss", loss, self.current_epoch)
        self.summary_writer.add_scalar("Rewards", self.reward_sum, self.current_epoch)
        self.summary_writer.add_scalar("Final Score", self.score, self.current_epoch)

        if self.score > self.high_score:
            self.high_score = self.score

        self.total_loss += loss
        self.reward_sum = 0
        self.current_epoch += 1

    def print_state(self, loss):
        sys.stdout.write('\r')
        sys.stdout.write(f"{str(self.current_epoch):>6} {str(self.score):>6} {str(self.high_score):>11.3} "
                         + f"{loss.item():>13.5f} {self.n_suicides / self.current_epoch:>13.5f}")