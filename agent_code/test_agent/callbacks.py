import os
import pickle
import random
import torch
from .model import LinearQNet
import torch, torch.nn as nn

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.n_features = 15 * 15 + 3
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = LinearQNet(self.n_features, 600, 6)
        nn.init.normal_(self.model.linear1.weight, mean=0, std=1.0)
        nn.init.normal_(self.model.linear2.weight, mean=0, std=1.0)
        nn.init.normal_(self.model.linear3.weight, mean=0, std=1.0)
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        self.model = LinearQNet(self.n_features, 300, 6)
        self.model.load_state_dict(torch.load('my-saved-model.pt'))
        self.model.eval()

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .5
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        state = torch.tensor(state_to_features(self, game_state), dtype=torch.float)
        prediction = self.model(state)
        action = ACTIONS[torch.argmax(prediction).item()]
        self.logger.debug("Querying model for action: {action}".format(action=action))
        return action

        # return np.random.choice(ACTIONS, p=self.model)


def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.
    
    game_state: {
    'round': int,
    'step': int,
    'field': np.array(width, height),
    'bombs': [((int, int), int), ...],
    'explosion_map': np.array(width, height),
    'coins': [(x, y), ...],
    'self': (str, int, bool, (int, int)),
    'others': [(str, int, bool, (int, int)), ...], }

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """

    # TODO: implement perceptual field with:
    #  own coordinates
    #  blocking walls as p x p,
    #  active flame blocks as p x p
    #  coordinates of coins 9 * (x,y). sort the coins by distance to our agent. Fill the rest of the coin matrix by (-1, -1)
    #  For the bombs, 4 x 3: for every bomb coordinate and countdown. Fill with (-1, -1, 0) as filler
    #  32 + 2p^2
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return np.zeros(self.n_features)

    map = game_state['field'][1:-1, 1:-1]
    # idea: positive values for "good" fields, negative values for danger
    for coin_coord in game_state['coins']:
        map[coin_coord[0]-1, coin_coord[1]-1] = 100
    for bomb in game_state['bombs']:
        bomb_coord = bomb[0]
        timer = bomb[1]
        # bombs detonate after four steps
        # idea: the danger from bomb gets larger and larger: -10, -20, -30
        map[bomb_coord[0]-1, bomb_coord[1]-1] = -10 - 10*(3-timer)
    # for how many more steps an explosion will be present
    # explosions linger for two time steps
    map[game_state['explosion_map'][1:-1, 1:-1] == 2] = -100
    map[game_state['explosion_map'][1:-1, 1:-1] == 1] = -90

    for opponent in game_state['others']:
        opp_coord = opponent[3]
        can_throw_bomb = opponent[2]
        map[opp_coord[0]-1, opp_coord[1]-1] = 50 - 5 * int(can_throw_bomb)

    map_vector = np.concatenate(map)

    self_coord = game_state['self'][3]
    can_throw_bomb = game_state['self'][2]
    state_vector = np.concatenate((map_vector, list(self_coord), [can_throw_bomb]))

    return state_vector # 15x15+3 entries
