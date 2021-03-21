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

# rewards are given only once for events:
MOVED_TOWARDS_CENTER_1, MOVED_TOWARDS_CENTER_2 = "MOVED_TOWARDS_CENTER_1", "MOVED_TOWARDS_CENTER_2"
MOVED_TOWARDS_CENTER_3, MOVED_TOWARDS_CENTER_4 = "MOVED_TOWARDS_CENTER_3", "MOVED_TOWARDS_CENTER_4"
MOVED_TOWARDS_CENTER_5, MOVED_TOWARDS_CENTER_6 = "MOVED_TOWARDS_CENTER_5", "MOVED_TOWARDS_CENTER_6"
REACHED_CENTER = "REACHED_CENTER"


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
    return events


def entered_dead_end_after_bombing(self_action, new_game_state, events):
    # if LAST action was BOMB:
    if len(new_game_state['bombs']) > 0 and new_game_state['bombs'][0][1] == 2:
        coord = new_game_state['self'][3]
        if self_action in ['UP', 'DOWN'] and e.INVALID_ACTION not in events:
            down = 1
            if self_action == 'UP':
                down = -1
            while True:
                can_not_escape_to_side = new_game_state['field'][coord[0]+1, coord[1] + down - np.sign(down)] in [1, -1] and \
                                         new_game_state['field'][coord[0]-1, coord[1] + down - np.sign(down)] in [1, -1]
                if can_not_escape_to_side:
                    can_not_escape_to_front = new_game_state['field'][coord[0], coord[1] + down] in [1, -1]
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
                can_not_escape_to_side = new_game_state['field'][coord[0] + right - np.sign(right), coord[1] + 1] in [1, -1] and \
                                         new_game_state['field'][coord[0] + right - np.sign(right), coord[1] - 1] in [1, -1]
                if can_not_escape_to_side:
                    can_not_escape_to_front = new_game_state['field'][coord[0] + right, coord[1]] in [1, -1]
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
        e.GOT_KILLED: 0,
        e.SURVIVED_ROUND: 0,
        e.OPPONENT_ELIMINATED: 0,

        e.BOMB_DROPPED: -50,
        # PLACED_BOMB_NEXT_TO_CRATE see below
        e.BOMB_EXPLODED: 0,
        DEAD_END: 0,
        e.KILLED_SELF: -100,
        e.KILLED_OPPONENT: 0,
        SURVIVED_OWN_BOMB: 0,
        e.CRATE_DESTROYED: 30,
        e.COIN_FOUND: 0,

        e.COIN_COLLECTED: 300,
        COLLECTED_THIRD_OR_HIGHER_COIN: 0,

        e.INVALID_ACTION: -50,
        PERFORMED_SAME_INVALID_ACTION_TWICE: 0,
        e.WAITED: -30,
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
