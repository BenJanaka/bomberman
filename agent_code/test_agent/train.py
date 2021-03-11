import random
import torch
import numpy as np
from collections import namedtuple, deque
from typing import List
import torch.nn as nn
import torch.optim as optim
import events as e
import sys
from .callbacks import state_to_features

from .model import LinearQNet

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 100  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 1000

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
SURVIVED_OWN_BOMB = "SURVIVED_OWN_BOMB"

actions_dic = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.learning_rate = 0.01
    self.gamma = 0.8
    self.model = LinearQNet(15*15+3, 300, 6) # input size, hidden size, output size
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
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
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # if old_game_state is not None:
    # self.oppenents = [old_game_state['others'] ]
    self.round = new_game_state["round"]
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # reward for placing a bomb and not running into its explosion
    # condition: alive + in old state agent was not able to drop a bomb but now is able to
    if self_action is not None and not old_game_state['self'][2] and new_game_state['self'][2]:
        events.append(SURVIVED_OWN_BOMB)

    t = Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state),
                   reward_from_events(self, events))
    # state_to_features is defined in callbacks.py
    self.transitions.append(t)

    state, action, next_state, reward = t.state, t.action, t.next_state, t.reward
    #if state is not None:
    train_step(self, [state], [action], [next_state], [reward])


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(
        Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))

    if len(self.transitions) > BATCH_SIZE:
        batch = random.sample(self.transitions, BATCH_SIZE)  # list of tuples
    else:
        batch = self.transitions

    states, actions, next_states, rewards = zip(*batch)

    # print("States at the end of a round: ")
    # print(states[1:])
    # print("Next-States at the end of a round: ")
    # print(list(next_states[1:]))
    # print("shape of states", states[1].shape)

    train_step(self, list(states), list(actions), list(next_states), list(rewards))

    # Store the model
    # TODO: Store best record
    # with open("my-saved-model.pt", "wb") as file:
    #     pickle.dump(self.model, file)
    self.model.save()

def Action_To_One_Hot(action):
    one_hot = np.zeros(6)
    # If we are at the start of the game, we simply set the current action to wait.
    if action is None:
        one_hot[4] = 1
        return one_hot
    else:
        one_hot[actions_dic[action]] = 1
        return one_hot


def train_step(self, state, action, next_state, reward):
    # print("reward before creating tensor", reward)
    # print("shape of transitions", self.transitions.count)
    # if state is None:
    #     return

    state = torch.tensor(state, dtype=torch.float)
    none_indices = [i for i in range(len(next_state)) if next_state[i] is None]
    done = np.zeros(len(next_state))
    done[none_indices] = True
    next_state = [np.zeros(self.n_features) if v is None else v for v in next_state]
    next_state = torch.tensor(next_state, dtype=torch.float)
    actions_one_hot = list(map(Action_To_One_Hot, action))
    action = torch.tensor(actions_one_hot)
    reward = torch.tensor(reward, dtype=torch.float)

    # 1: predicted Q values: expected reward of current state and action with dimension 6 (# actions)
    Q_pred = self.model(state)
    Q = Q_pred.clone()

    for idx in range(len(reward)):
        # if not next_state[idx] is None:
        # if not action[idx] is None: # final states have no action and we replaced None states with np.zeros(626)
        if done[idx]:
            Q_new = reward[idx]
        else:
            # temporal difference (TD) value estimation (see third lecture examples)
            Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # for the following approach, Q is would be a tensor of shape (state_size, action_size) which is too large to store
            # as there are a huge number of possible states
            # Q_new = Q_pred + self.learning_rate * (reward[idx] + self.gamma * torch.max(self.model(next_state[idx]) - Q_pred))

        Q[idx][torch.argmax(action[idx]).item()] = Q_new

    # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
    # Q_pred.clone()
    # preds[argmax(action)] = Q_new

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    self.optimizer.zero_grad()
    loss = self.criterion(Q, Q_pred)

    self.logger.info("Current Loss: {loss}".format(loss=loss))
    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()
    # Calling the step function on an Optimizer makes an update to its
    # parameters
    self.optimizer.step()


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        PLACEHOLDER_EVENT: -.1,  # idea: the custom event is bad

        SURVIVED_OWN_BOMB: 10,

        e.CRATE_DESTROYED: .5,
        e.KILLED_SELF: -500,
        e.GOT_KILLED: -10,
        e.INVALID_ACTION: -50,
        e.SURVIVED_ROUND: 0,

        e.WAITED: -50,
        e.BOMB_DROPPED: 0.1,
        e.BOMB_EXPLODED: 0,
        e.COIN_FOUND: 0,
        e.OPPONENT_ELIMINATED: 0.5,
        e.MOVED_LEFT: 0.1,
        e.MOVED_RIGHT: 0.1,
        e.MOVED_UP: 0.1,
        e.MOVED_DOWN: 0.1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
