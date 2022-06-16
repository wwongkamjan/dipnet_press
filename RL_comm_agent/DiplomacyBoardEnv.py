__authors__ = "Wichayaporn Wongkamjan"
__email__ = "wwongkam@umd.edu"

import ujson as json
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("../..")

from diplomacy.utils.export import to_saved_game_format
from diplomacy_research.models.gym.environment import DiplomacyEnv
from diplomacy_research.models.self_play.reward_functions import NormNbCentersReward, IntNormNbCentersReward
from diplomacy_research.models.state_space import extract_state_proto



class DiplomacyBoardEnv(DiplomacyEnv):

    def __init__(self):
        super().__init__(self)
        self.prev_state_proto = None
        self.output_dir = None

    def get_reward(self, power_name):
        """ return reward from last power's action, this function can be called after game.process"""
        prev_state_proto = self.prev_state_proto
        state_proto = extract_state_proto(self.game)
        is_terminal_state = self.is_done()
        done_reason = self.done_reason()
        
        # NormNbCentersReward for terminal state
        if is_terminal_state:
            reward_cls = NormNbCentersReward()
            return reward_cls.get_reward(prev_state_proto, state_proto, power_name, is_terminal_state, done_reason)
        else:
            reward_cls = IntNormNbCentersReward()
            return reward_cls.get_reward(prev_state_proto, state_proto, power_name, is_terminal_state, done_reason)

    def process(self):
        """ Requests that the game processes the current orders """
        self.prev_state_proto = extract_state_proto(self.game)
        self.game.process()
        current_phase = self.game.get_current_phase()
        if current_phase != 'COMPLETED':
            self._last_known_phase = current_phase

    def save_game(self):
        """ Save the buffer to disk """
        # Building saved game
        self.saved_game = to_saved_game_format(self.game)
        self.saved = True

        # Saving to disk
        if self.output_dir:
            timestamp = int(time.time())
            output_path = os.path.join(self.output_dir, '{}_{}.json'.format(timestamp, self.game_id))
            with open(output_path, 'w') as file:
                file.write(json.dumps(self.saved_game))

