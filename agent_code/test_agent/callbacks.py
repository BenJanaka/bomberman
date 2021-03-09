import os
import pickle
import random
import torch

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
    
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = 80 - game_state['round']
    if self.train and random.randint(0, 200) < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        state = torch.tensor(state_to_features(game_state), dtype=torch.float)
        prediction = self.model(state)
        return ACTIONS[torch.argmax(prediction).item()]
    
        # self.logger.debug("Querying model for action.")
        # return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.
    
    game_state: {'round': int,
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
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    bombs = np.zeros(12) # at most 4 bombs * 3 values
    for i, bomb in enumerate(game_state['bombs']):
        bombs[3*i: 3*i+3] = np.array([bomb[0][0], bomb[0][1], bomb[1]])
    
    # field_size = np.shape(game_state['field'])[0] * np.shape(game_state['field'])[1]
    coins = np.zeros(9 * 2) # there are 9 coins distributed
    for i, coin in enumerate(game_state['coins']):
        coins[2*i: 2*i+2] = np.array([coin[0], coin[1]])
    
    self = player_to_feature(game_state['self'])
    
    others = np.zeros(12) # at most 3 opponents
    for i, opponent in enumerate(game_state['others']):
        others[4*i: 4*i+4] = np.array([opponent[0], opponent[1]])
    
    state_list = np.array([game_state['round'], # 1
        game_state['step'], # 1
        game_state['field'].reshape(-1), # 17x17
        bombs, # 12
        game_state['explosion_map'].reshape(-1), # 17x17
        coins, # 9x2
        self, # 4
        others]) # 4x3
        
    return np.stack(state_list) # 626 entries in total
    
    # # For example, you could construct several channels of equal shape, ...
    # channels = []
    # channels.append(...)
    # # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # # and return them as a vector
    # return stacked_channels.reshape(-1)

def player_to_feature(SELF):
    """
    SELF: (str, int1, bool2, (int3, int4))
    returns: [int1, int2, int3, int4]
    The string holding the players name is dropped
    """
    return np.array([SELF[1], int(SELF[2]), SELF[3][0], SELF[3][1]])
