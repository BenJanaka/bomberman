import events as e
import numpy as np

SURVIVED_OWN_BOMB = "SURVIVED_OWN_BOMB"
COLLECTED_THIRD_OR_HIGHER_COIN = "COLLECTED_THIRD_OR_HIGHER_COIN"
PERFORMED_SAME_INVALID_ACTION_TWICE = "PERFORMED_SAME_INVALID_ACTION_TWICE"

# the 0 is replaced by the number of crates next to the bomb (3 at most)
PLACED_BOMB_NEXT_TO_CRATE = "PLACED_BOMB_NEXT_TO_CRATE_0"

# ran into a dead end right after bomb drop
# only valid if agent plays without opponents
DEAD_END = 'DEAD_END'
DEAD_END_BOMB_POSITION = "DEAD_END_BOMB_POSITION"

# rewards are given only once for events:
MOVED_TOWARDS_CENTER_1, MOVED_TOWARDS_CENTER_2 = "MOVED_TOWARDS_CENTER_1", "MOVED_TOWARDS_CENTER_2"
MOVED_TOWARDS_CENTER_3, MOVED_TOWARDS_CENTER_4 = "MOVED_TOWARDS_CENTER_3", "MOVED_TOWARDS_CENTER_4"
MOVED_TOWARDS_CENTER_5, MOVED_TOWARDS_CENTER_6 = "MOVED_TOWARDS_CENTER_5", "MOVED_TOWARDS_CENTER_6"
REACHED_CENTER = "REACHED_CENTER"
IN_DANGER = "IN_DANGER"
MOVED_TOWARDS_CLOSEST_COIN = 'MOVED_TOWARDS_CLOSEST_COIN'
MOVED_AWAY_FROM_CLOSEST_COIN = 'MOVED_AWAY_FROM_CLOSEST_COIN'

def append_events(self, old_game_state, self_action, new_game_state, events):
    # reward for placing a bomb and not running into its explosion
    # condition: alive + in old state agent was not able to drop a bomb but now is able to
    if self_action is not None and not old_game_state['self'][2] and new_game_state['self'][2]:
        events.append(SURVIVED_OWN_BOMB)
    n_left_coins = len(new_game_state['coins'])
    if "COIN_COLLECTED" in events and n_left_coins <= 6:
        events.append(COLLECTED_THIRD_OR_HIGHER_COIN)
    if len(self.transitions) > 2:
        last_transition = self.transitions[-1]
        if "INVALID_ACTION" in events and self_action == last_transition.action:
            events.append(PERFORMED_SAME_INVALID_ACTION_TWICE)
    if self_action == 'BOMB' and len(new_game_state['bombs']) > 0 and e.INVALID_ACTION not in events:
        bomb_coord = new_game_state['bombs'][0][0]
        n_crates = 0
        for position in [new_game_state['field'][bomb_coord[0] + 1, bomb_coord[1]],
                         new_game_state['field'][bomb_coord[0] - 1, bomb_coord[1]],
                         new_game_state['field'][bomb_coord[0], bomb_coord[1] + 1],
                         new_game_state['field'][bomb_coord[0], bomb_coord[1] - 1]]:
            if position == 1:
                n_crates += 1
        if n_crates > 0:
            PLACED_BOMB_NEXT_TO_CRATE = "PLACED_BOMB_NEXT_TO_CRATE_" + str(int(n_crates))
            events.append(PLACED_BOMB_NEXT_TO_CRATE)
    if self_action == "BOMB" and e.INVALID_ACTION not in events:
        if not can_escape_bomb(new_game_state):
            events.append(DEAD_END_BOMB_POSITION)
    if entered_dead_end_after_bombing(self_action, new_game_state, events):
        events.append(DEAD_END)

    # if agents moved towards center:
    # closest point is saved so that a repeated back and forth movement is prevented
    self_coord = new_game_state['self'][3]
    if abs(self_coord[0] - 8) < self.closest_to_center and abs(self_coord[1] - 8) < self.closest_to_center:
        if self.closest_to_center == 6:
            events.append(MOVED_TOWARDS_CENTER_6)
        elif self.closest_to_center == 5:
            events.append(MOVED_TOWARDS_CENTER_5)
        elif self.closest_to_center == 4:
            events.append(MOVED_TOWARDS_CENTER_4)
        elif self.closest_to_center == 3:
            events.append(MOVED_TOWARDS_CENTER_3)
        elif self.closest_to_center == 2:
            events.append(MOVED_TOWARDS_CENTER_2)
        elif self.closest_to_center == 1:
            events.append(MOVED_TOWARDS_CENTER_1)
        elif self.closest_to_center == 0:
            events.append(REACHED_CENTER)

    # check if our agent is in dangerous area
    field = new_game_state['field']
    player_id = 42
    field[new_game_state['self'][3][0]][new_game_state['self'][3][1]] = player_id
    field = np.pad(field, pad_width=2, mode='constant', constant_values=-1)
    in_danger = False
    for bomb in new_game_state['bombs']:
        bomb_coord = np.array(bomb[0])
        if bomb_coord[0] == new_game_state['self'][3][0] and bomb_coord[1] == new_game_state['self'][3][1]:
            in_danger = True
            break
        else:
            bomb_coord = bomb_coord + [2, 2]
            t = np.ones((7, 7))
            t[:, 3] = 0
            t[3, :] = 0
            explosion_matrix = field[bomb_coord[0] - 3:bomb_coord[0] + 4, bomb_coord[1] - 3:bomb_coord[1] + 4]
            is_dangerous = (explosion_matrix == t).astype(int)
            is_dangerous[is_dangerous == 0] = -1
            in_danger = (field[bomb_coord[0] - 3:bomb_coord[0] + 4,
                         bomb_coord[1] - 3:bomb_coord[1] + 4] == player_id) == is_dangerous
            in_danger = np.any(in_danger)
    if in_danger:
        events.append(IN_DANGER)


    # Check if agent moved to the closest coin on the field
    own_pos = np.array(new_game_state['self'][3])

    old_coins = old_game_state['coins']
    new_coins = new_game_state['coins']

    if old_coins and new_coins:

        old_distances = np.linalg.norm(own_pos - old_coins, axis=1)
        old_nearest = old_distances.argmin()

        new_distances = np.linalg.norm(own_pos - new_coins, axis=1)
        new_nearest = new_distances.argmin()

        if new_nearest == old_nearest:
            if new_distances[new_nearest] < old_distances[old_nearest]:
                events.append(MOVED_TOWARDS_CLOSEST_COIN)
            elif new_distances[new_nearest] > old_distances[old_nearest]:
                events.append(MOVED_AWAY_FROM_CLOSEST_COIN)
    return events


def can_escape_bomb(new_game_state):
    test_game_state = new_game_state.copy()
    actions = ['DOWN', 'UP', 'RIGHT', 'LEFT']
    for i, movement in enumerate([[0, 1], [0, -1], [1, 0], [-1, 0]]):
        x, y = new_game_state['self'][3][0] + movement[0], new_game_state['self'][3][1] + movement[1]
        test_game_state['self'] = (new_game_state['self'][0], new_game_state['self'][1], new_game_state['self'][2], (x, y))
        if new_game_state['field'][test_game_state['self'][3]] == 0: # move is valid
            timer = new_game_state['bombs'][0][1] - 1
            test_game_state['bombs'] = [(new_game_state['bombs'][0][0], timer)]
            if not entered_dead_end_after_bombing(actions[i], test_game_state, []):
                return True
    return False


def entered_dead_end_after_bombing(self_action, state, events):
    assert len(state['bombs']) <= 1, "More than one bomb. Turn off reward for dead ends."
    # if LAST action was BOMB:
    if len(state['bombs']) > 0 and state['bombs'][0][1] == 2:
        coord = state['self'][3]
        if self_action in ['UP', 'DOWN'] and e.INVALID_ACTION not in events:
            down = 1
            if self_action == 'UP':
                down = -1
            while True:
                can_not_escape_to_side = state['field'][coord[0]+1, coord[1] + down - np.sign(down)] in [1, -1] and \
                                         state['field'][coord[0]-1, coord[1] + down - np.sign(down)] in [1, -1]
                if can_not_escape_to_side:
                    can_not_escape_to_front = state['field'][coord[0], coord[1] + down] in [1, -1]
                    if can_not_escape_to_front:
                        return True
                    else:
                        down += np.sign(down)
                        if abs(down) > 3:
                            break
                else:
                    break
        elif self_action in ['RIGHT', 'LEFT'] and e.INVALID_ACTION not in events:
            right = 1
            if self_action == 'LEFT':
                right = -1
            while True:
                can_not_escape_to_side = state['field'][coord[0] + right - np.sign(right), coord[1] + 1] in [1, -1] and \
                                         state['field'][coord[0] + right - np.sign(right), coord[1] - 1] in [1, -1]
                if can_not_escape_to_side:
                    can_not_escape_to_front = state['field'][coord[0] + right, coord[1]] in [1, -1]
                    if can_not_escape_to_front:
                        return True
                    else:
                        right += np.sign(right)
                        if abs(right) > 3:
                            break
                else:
                    break
    else:
        return False


def reward_from_events(self, events):
    """
    Rewards your agent get so as to en/discourage certain behavior.
    """
    game_rewards = {
        e.GOT_KILLED: -200,
        e.SURVIVED_ROUND: 0,
        e.OPPONENT_ELIMINATED: 0,

        e.BOMB_DROPPED: -3,
        DEAD_END_BOMB_POSITION: -200,
        DEAD_END: -30,
        # PLACED_BOMB_NEXT_TO_CRATE see below
        e.BOMB_EXPLODED: 0,
        e.KILLED_SELF: -500,
        e.KILLED_OPPONENT: 200,
        SURVIVED_OWN_BOMB: 5,
        e.CRATE_DESTROYED: 20,
        e.COIN_FOUND: 0,

        e.COIN_COLLECTED: 100,
        COLLECTED_THIRD_OR_HIGHER_COIN: 0,

        e.INVALID_ACTION: -10,
        PERFORMED_SAME_INVALID_ACTION_TWICE: 0,
        e.WAITED: -10,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,

        # rewards only make sense if agent starts at field edge:
        # MOVED_TOWARDS_CENTER_6: 10,
        # MOVED_TOWARDS_CENTER_5: 11,
        # MOVED_TOWARDS_CENTER_4: 12,
        # MOVED_TOWARDS_CENTER_3: 13,
        # MOVED_TOWARDS_CENTER_2: 14,
        # MOVED_TOWARDS_CENTER_1: 15,
        # REACHED_CENTER: 16,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        if "PLACED_BOMB_NEXT_TO_CRATE" in event:
            n_crates = int(event[-1])
            reward_sum += 50 + 5 * n_crates
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
