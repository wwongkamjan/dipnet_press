# trainer for new A2C 
from pytorch_DRL.A2C import A2C
from pytorch_DRL.common.utils import ma_agg_double_list, dict_to_arr, arr_dict_to_arr, index_to_one_hot
# A2C: https://github.com/ChenglongChen/pytorch-DRL/
from tornado import gen
import ujson as json
import csv
from test_proposal_MAA2C import test

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("../..")
sys.path.append("../../baseline_bots")
import numpy as np
import matplotlib.pyplot as plt
import random

from diplomacy import Game, Message
from DiplomacyBoardEnv import DiplomacyBoardEnv
from diplomacy.engine.message import Message
from diplomacy.utils.export import to_saved_game_format
from diplomacy_research.utils.cluster import start_io_loop, stop_io_loop
from DAIDE import ORR, XDO
from baseline_bots.bots.pushover_bot import PushoverBot
from baseline_bots.bots.random_no_press import RandomNoPressBot
from baseline_bots.bots.dipnet.no_press_bot import NoPressDipBot

from diplomacy_research.models.order_based.v015_film_transformer_gpt.model import PolicyModel
from diplomacy_research.models.state_space import extract_state_proto
from diplomacy_research.utils.tensorflow import tf

def main():    

    policymodel = PolicyModel()
    env = DiplomacyBoardEnv()
    env.game = Game()
    print('game state shape:', tf.shape(env.game.get_state()))
    encoded_board_state = policymodel._encode_board(env.game.get_state(), env.game.map)
    print('encoded game state shape:', tf.shape(encoded_board_state))
        
if __name__ == '__main__':
    main()