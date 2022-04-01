import gym
import numpy as np
from tornado import gen
from diplomacy import Game
from diplomacy.engine.message import Message
from diplomacy.utils.export import to_saved_game_format
from diplomacy_research.players.benchmark_player import DipNetSLPlayer
from diplomacy_research.utils.cluster import start_io_loop, stop_io_loop
from diplomacy_research.models.state_space import get_order_tokens
from bot_communication import Diplomacy_Press, Diplomacy_Press_Player
from MAA2C import MAA2C
from common.utils import ma_agg_double_list

class DiplomacyEnv(gym.Env):
  def __init__(self):
    self.done = False
    self.n_agents = 1
    self.agent_id = [id for id in range(n_agents)]
    self.power_onehot = {}
    self.order_onehot = {}
    # stance vector of [power1][power2],  
    # orders:
    #         unit's power, England= ... one hot for 7 powers = [Eng, France, ..]
    #         type of order, one hot [move, hold, support, attack, convoy]
    #         attack whom/move to whose territory one hot for 7 powers = [Eng, France, ..]
    # cur_obs for each agent and action for each agent
    self.observation_space = gym.spaces.Box(low=np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                                            high=np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
                                            dtype=np.float32)
    self.action_space = gym.spaces.Discrete(2) # 
    self.cur_obs = None
    self.episode_len = 0
    self.dip_net = None
    self.dip_game = None

    
  def reset(self): 
    # return to initial state - Diplomacy game and DipNet reset
    # get stance vector, orders from framework
    self.dip_player = DipNetSLPlayer()
#   dip_player =  Diplomacy_Press_Player(Player=random_player())
    self.dip_game =  Game()
    self.dip_player.init_communication(self.dip_game.game.powers)
    self.episode_len = 0
    self.cur_obs = {agent_id: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for agent_id in self.agent_id} #neutral for any power and no order
    return self.cur_obs
    
  def step(self, action): # return state, reward, done, info
      
    if self.dip_game.is_game_done:
      self.done = True
      reward = 0
    reward = get from next phase result (0/+1/-1 get/lose supply center)
    return None, 
      
    

