import math
import os
import pickle
import random
import torch
from .model import LinearQNet
import torch, torch.nn as nn
import numpy as np

VIEW_DIST = 14
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def create_model(self):
    model = LinearQNet(6).to(self.device)
    # for layer in model.children():
    #     if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
    #         layer.bias.data.fill_(0.)
    #         nn.init.normal_(layer.weight, mean=0., std=1. / 100)
    return model


def load_model(self):
    model = LinearQNet(6).to(self.device)
    self.saved_state = model.load(self.path)
    model.load_state_dict(self.saved_state['model'])
    return model


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
    self.overwrite = True
    self.path = "my-saved-model.pt"
    self.view_dist = VIEW_DIST
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    self.logger.info("running on device {device}".format(device=self.device.type))
    if not os.path.isfile(self.path) or self.overwrite:
        self.logger.info("Setting up model from scratch.")
        self.model = create_model(self)
    else:
        self.logger.info("Loading model from saved state.")
        self.model = load_model(self)
        self.logger.info("Loaded high score: {score}".format(score=self.saved_state['score']), )

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
    if self.train and random.random() < self.hpm.exploration_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.15, .15, .15, .15, .2, .2])
    else:
        state = torch.tensor(state_to_features(self, game_state), dtype=torch.float)
        prediction = self.model(state.to(self.device))
        action = ACTIONS[torch.argmax(prediction).item()]
        self.logger.debug("Querying model for action: {action}".format(action=action))
        return action


def state_to_features(self, game_state):
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
    assert game_state is not None, "Game state is None"

    self_coord = list(game_state["self"][3])
    # Walls 5x5 around our agent: self.view_dist = 2
    shift = self.view_dist - 1
    left = self_coord[0] + shift - self.view_dist
    right = self_coord[0] + shift + self.view_dist
    bottom = self_coord[1] + shift + self.view_dist
    top = self_coord[1] + shift - self.view_dist

    padded_field = np.pad(game_state['field'], self.view_dist - 1, constant_values=0).astype(np.float64)
    crates = np.zeros(np.shape(padded_field))
    crates[padded_field == 1] = 1
    crates = crates[left:right + 1, top:bottom + 1]
    walls = np.zeros(np.shape(padded_field))
    walls[padded_field == -1] = 1
    walls = walls[left:right + 1, top:bottom + 1]

    explosions = np.pad(game_state['explosion_map'], self.view_dist - 1, constant_values=0)
    # here we exploit that explosions are only one time step lethal
    explosions[explosions == 1] = 0
    explosions = explosions[left:right + 1, top:bottom + 1]

    bombs = np.zeros(np.shape(padded_field))
    # add pre-explosions for each bomb if no crate is on field with timer
    power = 3
    for bomb in game_state["bombs"]:
        x, y = bomb[0][0] + shift, bomb[0][1] + shift
        timer = bomb[1]
        bombs[x, y] = - 30 - (4 - timer) * 5
        for i in range(1, power + 1):
            if bombs[x + i, y] == -1:
                break
            if bombs[x + i, y] != 1:
                bombs[x + i, y] = (-30 + i * 5.) - (4 - timer) * 5

        for i in range(1, power + 1):
            if bombs[x - i, y] == -1:
                break
            if bombs[x - i, y] != 1:
                bombs[x - i, y] = (-30 + i * 5.) - (4 - timer) * 5

        for i in range(1, power + 1):
            if bombs[x, y + i] == -1:
                break
            if bombs[x, y + i] != 1:
                bombs[x, y + i] = (-30 + i * 5.) - (4 - timer) * 5

        for i in range(1, power + 1):
            if bombs[x, y - i] == -1:
                break
            if bombs[x, y - i] != 1:
                bombs[x, y - i] = (-30 + i * 5.) - (4 - timer) * 5
    bombs = bombs[left:right + 1, top:bottom + 1]

    coins = np.zeros(np.shape(padded_field))
    for coin in game_state["coins"]:
        coins[coin[0] + shift, coin[1] + shift] = 1
    coins = coins[left:right + 1, top:bottom + 1]

    bomb_ready = int(game_state['self'][2])
    player = np.zeros(np.shape(padded_field))
    player[self_coord[0] + shift, self_coord[1] + shift] = - bomb_ready - 1
    player = player[left:right + 1, top:bottom + 1]

    opponents = np.zeros(np.shape(padded_field))
    for opponent in game_state['others']:
        opponent_coord = opponent[3]
        opponents[opponent_coord[0] + shift, opponent_coord[1] + shift] = opponent[2] + 1
    opponents = opponents[left:right + 1, top:bottom + 1]
    return np.stack([walls, crates, coins, bombs, explosions, player, opponents])
