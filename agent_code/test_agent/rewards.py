import events as e

SURVIVED_OWN_BOMB = "SURVIVED_OWN_BOMB"
COLLECTED_THIRD_OR_HIGHER_COIN = "COLLECTED_THIRD_OR_HIGHER_COIN"
PERFORMED_SAME_INVALID_ACTION_TWICE = "PERFORMED_SAME_INVALID_ACTION_TWICE"

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


def reward_from_events(self, events):
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.GOT_KILLED: -10,
        e.SURVIVED_ROUND: 0,
        e.OPPONENT_ELIMINATED: 0,

        e.BOMB_DROPPED: -15,
        e.BOMB_EXPLODED: 0,
        e.KILLED_SELF: -50,
        e.KILLED_OPPONENT: 200,
        SURVIVED_OWN_BOMB: 20,
        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 0,

        e.COIN_COLLECTED: 50,
        COLLECTED_THIRD_OR_HIGHER_COIN: 10,

        e.INVALID_ACTION: -10,
        PERFORMED_SAME_INVALID_ACTION_TWICE: -10,
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
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
