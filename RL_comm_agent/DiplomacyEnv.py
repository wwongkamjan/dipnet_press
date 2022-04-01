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
    self.done = False
    # stance vector of [power1][power2],  
    # orders:
    #         unit's power, England= ... one hot for 7 powers = [Eng, France, ..]
    #         type of order, one hot [move, hold, support, attack, convoy]
    #         attack whom/move to whose territory one hot for 7 powers = [Eng, France, ..]
    self.observation_space = gym.spaces.Box(low=np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                                            high=np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
                                            dtype=np.float32)
    self.action_space = gym.spaces.Discrete(2) # 
    self.cur_obs = None
    self.episode_len = 0
    
  def reset(self): 
    # return to initial state - Diplomacy game and DipNet reset
    # get stance vector, orders from framework
    self.episode_len = 0
    self.cur_obs =
  def step(self, action): # return state, reward, done, info

