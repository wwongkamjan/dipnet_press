# env - s,a,r to interact with agent 
# agent - network A2C
# trainer - train agent
# utils
# ref: 
import gym
from tornado import gen
import ujson as json
from diplomacy import Game
from diplomacy.engine.message import Message
from diplomacy.utils.export import to_saved_game_format
from diplomacy_research.players.benchmark_player import DipNetSLPlayer
from diplomacy_research.utils.cluster import start_io_loop, stop_io_loop
from diplomacy_research.models.state_space import get_order_tokens
from diplomacy.server.server_game import ServerGame
from diplomacy.daide.requests import RequestBuilder
from MAA2C import MAA2C
from common.utils import ma_agg_double_list

class DiplomacyEnv(gym.Env):
  def __init__(self):
    self.done = set()
    # stance vector of [power1][power2], neutral=0 ally=1 enemy=2 + 
    # orders:
    #         unit's power, England= ...
    #         type of order, move=0 support=1 
    self.observation_space = gym.spaces.MultiDiscrete([3, ])
    self.action_space = 
  def reset(self): 
    # return to initial state - Diplomacy game and DipNet reset
  def step(self, action): # return state, reward, done, info

