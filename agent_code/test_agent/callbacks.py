import os
import pickle
import random
import torch
from .model import LinearQNet
import torch, torch.nn as nn
import numpy as np

VIEW_DIST = 16
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_PROBS = [.15, .15, .15, .15, .2, .2]


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
    self.overwrite = False
    self.path = "my-saved-model.pt"
    self.view_dist = VIEW_DIST

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.logger.info("running on device {device}".format(device=self.device.type))

    if not os.path.isfile(self.path) or self.overwrite:
        self.logger.info("Setting up model from scratch.")
        self.model = LinearQNet(6).to(self.device)
        for layer in self.model.children():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                layer.bias.data.fill_(0.)
                nn.init.normal_(layer.weight, mean=0., std=1./100)

    else:
        self.logger.info("Loading model from saved state.")
        self.model = LinearQNet(6).to(self.device)
        self.saved_state = self.model.load(self.path)
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

    assert game_state is not None, "Game state is None"

    # Exploration vs exploitation
    if self.train and random.random() < self.exploration_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=ACTION_PROBS)
    else:
        state = torch.tensor(state_to_features(self, game_state), dtype=torch.float)
        prediction = self.model(state.to(self.device))
        action = ACTIONS[torch.argmax(prediction).item()]
        self.logger.debug("Querying model for action: {action}".format(action=action))
        return action


def state_to_features(self, game_state):
    assert game_state is not None, "Game state is None"

    self_coord = list(game_state["self"][3])
    # Walls 5x5 around our agent: self.view_dist = 2
    shift = self.view_dist-1
    left = self_coord[0]+shift - self.view_dist
    right = self_coord[0]+shift + self.view_dist
    bottom = self_coord[1]+shift + self.view_dist
    top = self_coord[1]+shift - self.view_dist

    padded_field = np.pad(game_state['field'], self.view_dist-1, constant_values=0).astype(np.float64)
    walls = padded_field
    explosions = np.pad(game_state['explosion_map'], self.view_dist-1, constant_values=0)
    # add pre-explosions for each bomb if no crate is on field with timer
    power = 3
    for bomb in game_state["bombs"]:
        x, y = bomb[0][0] + shift, bomb[0][1] + shift
        timer = bomb[1]
        walls[x, y] = - 30 - (4-timer) * 5

        for i in range(1, power + 1):
            if walls[x + i, y] == -1:
                break
            if walls[x + i, y] != 1:
                walls[x + i, y] = (-30 + i * 5.) - (4-timer) * 5

        for i in range(1, power + 1):
            if walls[x - i, y] == -1:
                break
            if walls[x - i, y] != 1:
                walls[x - i, y] = (-30 + i * 5.) - (4-timer) * 5

        for i in range(1, power + 1):
            if walls[x, y + i] == -1:
                break
            if walls[x, y + i] != 1:
                walls[x, y + i] = (-30 + i * 5.) - (4-timer) * 5

        for i in range(1, power + 1):
            if walls[x, y - i] == -1:
                break
            if walls[x, y - i] != 1:
                walls[x, y - i] = (-30 + i * 5.) - (4-timer) * 5

    coins = np.zeros(np.shape(padded_field))
    for coin in game_state["coins"]:
        coins[coin[0]+shift, coin[1]+shift] = 1

    # make crates to -1 so that agent knows not to run against them
    # and add crates to coin field
    coins[walls == 1] = -0.1
    coins = coins[left:right + 1, top:bottom + 1]

    # here we exploit that explosions are only one time step lethal
    explosions[explosions == 1] = 0
    walls -= 30 * explosions
    walls = walls[left:right + 1, top:bottom + 1]

    bomb_ready = int(game_state['self'][2])
    players = np.zeros(np.shape(padded_field))
    players[self_coord[0] + shift, self_coord[1] + shift] = - bomb_ready - 1
    for opponent in game_state['others']:
        opponent_coord = opponent[3]
        players[opponent_coord[0] + shift, opponent_coord[1] + shift] = opponent[2] + 1
    players = players[left:right + 1, top:bottom + 1]
    return np.stack([walls, coins, players])


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
    assert game_state is not None, "Game state is None"

    own_agent = [game_state["self"][2]] + list(game_state["self"][3])
    # Walls and explosion sub-fields. 3x3 around our agent
    walls_pxp = list(game_state['field'][own_agent[1]-1:own_agent[1]+2, own_agent[2]-1:own_agent[2]+2].flatten())
    explosions_pxp = list(game_state['explosion_map'][own_agent[1]-1:own_agent[1]+2, own_agent[2]-1:own_agent[2]+2].flatten())
    # list of coins where each coins coordinate is saved.
    coin_dist = [-1] * 9 * 2
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

    # Add each coin to the output list: coin_dist
    for idx, coin in enumerate(game_state['coins']):
        coin_dist[idx * 2: idx * 2 + 2] = list(own_pos - np.array(coin))

    # Add all bombs to the output list: bomb_list
    for idx, bomb in enumerate(state_bombs):
        # (x, y, countdown)
        # replace slice by actual values
        bomb_list[idx * 3:idx * 3 + 3] = list(bomb[0]) + [bomb[1]]

    # TODO: coordinates relative to agent + opponents
    # output = np.array(walls_pxp + explosions_pxp + coin_dist + bomb_list + own_agent, dtype=np.float32)
    output = np.array(walls_pxp + coin_dist + list(np.zeros(24)), dtype=np.float32)
    return output # list of length 51


def sortByDistance(element, distances):
    pass
