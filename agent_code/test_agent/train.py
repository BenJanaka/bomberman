import random
import torch
import numpy as np
from collections import namedtuple, deque
from typing import List
import torch.nn as nn
import torch.optim as optim
import sys
import os
from .callbacks import state_to_features, create_model
from .rewards import append_events, reward_from_events
from .plot import plot, update_plot_data, init_plot_data
from torch.utils.tensorboard import SummaryWriter
from .hyper_parameter_manager import HyperParameterManager
from .tensorboard_manager import TensorBoardManager

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 50000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 100
EXPLORATION_PROB = 0.05
LEARNING_RATE = 0.00002
EPSILON_DECAY = 0.0001
GAMMA = 0.99
TAU = 4

actions_dic = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}
MODE = 'HP-TEST'
N_TEST_EPOCHS = 300

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.run_id = 0
    self.hpm = HyperParameterManager(
        transition_history_size=TRANSITION_HISTORY_SIZE,
        batch_size=BATCH_SIZE,
        epsilon_decay=EPSILON_DECAY,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        tau=TAU,
        mode=MODE
    )

    self.tbm = TensorBoardManager(self.hpm, 'runs/300_master/')

    self.closest_to_center = 7
    # init_plot_data(self)
    # self.n_suicides = 0

    self.optimizer = optim.Adam(self.model.parameters(), lr=self.hpm.learning_rate)
    # Load the saved optimizer to continue training
    if not self.overwrite and os.path.isfile(self.path):
        self.optimizer.load_state_dict(self.saved_state['optimizer'])
        self.high_score = self.saved_state['score']
        print('Resumed training of loaded model from:', self.path)
    else:
        print('Started training session from scratch')
    print(f'with learning rate {self.hpm.learning_rate} and exploration probability {self.hpm.exploration_prob}')
    print(' round  score  high score          loss  suicide rate')

    # self.criterion = nn.SmoothL1Loss()
    self.criterion = nn.MSELoss()
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    SHORT TERM MEMORY
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from `old_game_state` to `new_game_state`
    """

    self.round = new_game_state["round"]
    self.step = new_game_state["step"]

    # in the first step, old state is None
    if old_game_state is None:
        return

    events = append_events(self, old_game_state, self_action, new_game_state, events)
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    t = Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state),
                   reward_from_events(self, events))
    self.transitions.append(t)

    state, action, next_state, reward = t.state, t.action, t.next_state, t.reward

    self.tbm.reward_sum += reward
    self.tbm.score = new_game_state['self'][1]

    train_step(self, [state], [action], [next_state], [reward])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    LONG TERM MEMORY
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param
    self: The same object that is passed to all of your callbacks.
    last_game_state:
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))

    if len(self.transitions) > BATCH_SIZE:
        batch = random.sample(self.transitions, BATCH_SIZE)  # list of tuples
    else:
        batch = self.transitions

    states, actions, next_states, rewards = zip(*batch)

    loss = train_step(self, list(states), list(actions), list(next_states), list(rewards))

    # save_model_if_mean_rewards_increased(self)

    self.tbm.add_plot_data(loss)

    # reset the model if the number of games for this test case is reached
    if self.tbm.current_epoch > N_TEST_EPOCHS:
        self.model = create_model(self)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hpm.learning_rate)
        self.tbm.prepare_next_training_instance()
        self.transitions = []

    if "KILLED_SELF" in events:
        self.tbm.n_suicides += 1

    # update_plot_data(self, len(batch))
    if self.tbm.current_epoch % 10 == 0:
        self.model.save(self.optimizer, self.tbm.high_score, self.path)
        # plot(self)

        # exploit this function for scheduled exploration prob
        self.hpm.exploration_prob = max(0.1, self.hpm.exploration_prob - self.hpm.epsilon_decay)


    self.closest_to_center = 7


def train_step(self, state, action, next_state, reward):
    state = torch.tensor(state, dtype=torch.float)
    none_indices = [i for i in range(len(next_state)) if next_state[i] is None]
    done = np.zeros(len(next_state))
    done[none_indices] = True
    next_state = [np.zeros(state[0].shape) if v is None else v for v in next_state]
    next_state = torch.tensor(next_state, dtype=torch.float)
    actions_one_hot = list(map(Action_To_One_Hot, action))
    action = torch.tensor(actions_one_hot)
    reward = torch.tensor(reward, dtype=torch.float)

    # predicted Q values: expected reward of current state and action with dimension (batch size, # actions)
    # update with temporal difference (TD) Q-learning algorithm (third lecture examples)
    Q_pred = self.model(state.to(self.device))
    Q = Q_pred.clone()

    for idx in range(len(reward)):
        if done[idx]:
            Q_new = reward[idx]
        else:
            Q_new = reward[idx] + self.hpm.gamma * torch.max(self.model(next_state[idx].to(self.device)))
        Q[idx][torch.argmax(action[idx]).item()] = Q_new

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    self.optimizer.zero_grad()
    loss = self.criterion(Q, Q_pred)
    self.tbm.loss_sum += loss.detach() * len(state)
    self.logger.info("Current Loss: {loss}".format(loss=loss))

    self.tbm.print_state(loss)

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # Calling the step function on an Optimizer makes an update to its parameters
    self.optimizer.step()

    return loss


def Action_To_One_Hot(action):
    one_hot = np.zeros(6)
    # If we are at the start of the game, we simply set the current action to wait.
    if action is None:
        one_hot[4] = 1
    else:
        one_hot[actions_dic[action]] = 1
    return one_hot


def save_model_if_mean_rewards_increased(self):
    # besser wenn alles gespeichert wird
    # score as sum of rewards
    games_to_average = 10
    if len(self.transitions) > games_to_average:
        # loop through transitions back to front
        idx = len(self.transitions) - 2
        score = 0
        game_count = 0
        # Take the average of the last 50 games as the score of our model.
        while game_count < games_to_average:
            is_end_of_game = self.transitions[idx].next_state is None
            if is_end_of_game:
                game_count = game_count + 1
            if idx < 0:
                break
            score = score + self.transitions[idx].reward
            idx = idx - 1
        score = score / game_count
        if score >= self.high_score_:
            self.high_score_ = score
            self.model.save(self.optimizer, score, self.path)
