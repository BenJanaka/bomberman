import os
import pickle
import random
import torch
from .model import LinearQNet
import torch, torch.nn as nn

import numpy as np

VIEW_DIST = 2
# N_FEATURES = 4*(VIEW_DIST*2+1)**2 + 1
N_FEATURES = 2 * (VIEW_DIST * 2 + 1) ** 2
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
    self.view_dist = VIEW_DIST
    self.n_features = N_FEATURES
    self.overwrite = False

    if not os.path.isfile("my-saved-model.pt") or self.overwrite:
        self.logger.info("Setting up model from scratch.")
        self.model = LinearQNet(self.n_features, 6)
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                # layer.bias.data.fill_(0.)
                nn.init.normal_(layer.weight, mean=0., std=1./6)

        self.model.train()
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()

    else:
        self.logger.info("Loading model from saved state.")
        self.model = LinearQNet(self.n_features, 6)
        self.saved_state = self.model.load()
        self.model.load_state_dict(self.saved_state['model'])
        self.logger.info("Loaded highscore: {score}".format(score=self.saved_state['score']), )

        if self.train:
            self.model.train()
        else:
            self.model.eval()



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if game_state is None:
        return "WAIT"
    # Exploration vs exploitation
    if self.train and random.random() < self.exploration_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .195, .005])
    else:
        state = torch.tensor(state_to_features(self, game_state), dtype=torch.float)
        prediction = self.model(state)
        action = ACTIONS[torch.argmax(prediction).item()]
        self.logger.debug("Querying model for action: {action}".format(action=action))
        print(prediction)
        print(action)
        return action

        # return np.random.choice(ACTIONS, p=self.model)

def state_to_features(self, game_state):
    if game_state is None:
        return np.zeros(self.n_features)

    self_coord = list(game_state["self"][3])
    # Walls 5x5 around our agent: self.view_dist = 2
    shift = self.view_dist-1
    left = self_coord[0]+shift - self.view_dist
    right = self_coord[0]+shift + self.view_dist
    bottom = self_coord[1]+shift + self.view_dist
    top = self_coord[1]+shift - self.view_dist

    padded_field = np.pad(game_state['field'], self.view_dist-1, constant_values=-1)
    walls = padded_field[left:right+1, top:bottom+1]
    padded_explosion_map = np.pad(game_state['explosion_map'], self.view_dist-1, constant_values=-1)
    explosions = padded_explosion_map[left:right + 1, top:bottom + 1]

    coins = np.zeros(np.shape(padded_field))
    for coin in game_state["coins"]:
        coins[coin[0]+shift, coin[1]+shift] = 1
    coins = coins[left:right+1, top:bottom+1]

    bombs = np.zeros(np.shape(padded_field))
    for bomb in game_state["bombs"]:
        bombs[bomb[0][0]+shift, bomb[0][1]+shift] = bomb[1]
    bombs = bombs[left:right+1, top:bottom+1]

    # TODO opponents
    bomb_ready = int(game_state['self'][2])

    output = np.stack([walls, explosions, coins, bombs]).flatten()
    output = np.concatenate((output, [bomb_ready]))
    # return output # list of length 51
    return np.stack([walls, coins]).flatten()


def state_to_features_(self, game_state: dict) -> np.array:
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
    #  own coordinates + bomb available = 3
    #  blocking walls as p x p = 9
    #  active flame blocks as = 9
    #  coordinates of coins 9 * (x,y). sort the coins by distance to our agent. Fill the rest of the coin matrix by (-1, -1) = 18
    #  For the bombs, 4 x 3: for every bomb coordinate and countdown. Fill with (-1, -1, 0) as filler
    #  33 + 2p^2
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return np.zeros(self.n_features)

    own_agent = [game_state["self"][2]] + list(game_state["self"][3])
    # Walls and explosion sub-fields. 3x3 around our agent
    walls_pxp = list(game_state['field'][own_agent[1]-1:own_agent[1]+2, own_agent[2]-1:own_agent[2]+2].flatten())
    explosions_pxp = list(game_state['explosion_map'][own_agent[1]-1:own_agent[1]+2, own_agent[2]-1:own_agent[2]+2].flatten())
    # list of coins where each coins coordinate is saved.
    coin_list = [-1] * 9 * 2
    # List of bombs where each bombs coordinate and its countdown in listed.
    bomb_list = [-1] * 3 * 4  # 3 * 9

    # Create distances between own position and bomb positions
    own_pos = np.array((own_agent[1], own_agent[2]))
    state_bombs = game_state['bombs']
    distances = []
    for idx, bomb in enumerate(state_bombs):
        bomb_pos = np.array((bomb[0][0], bomb[0][1]))
        # distance between own and bomb
        distance = np.linalg.norm(own_pos - bomb_pos)
        distances.append(distance)

    # Add each coin to the output list: coin_list
    for idx, coin in enumerate(game_state['coins']):
        coin_list[idx * 2: idx * 2 + 2] = list(coin)

    # Add all bombs to the output list: bomb_list
    for idx, bomb in enumerate(state_bombs):
        # (x, y, countdown)
        # replace slice by actual values
        bomb_list[idx * 3:idx * 3 + 3] = list(bomb[0]) + [bomb[1]]

    output = np.array(own_agent + walls_pxp + explosions_pxp + coin_list + bomb_list, dtype=np.float32)
    return output # list of length 51




def sortByDistance(element, distances):
    pass
