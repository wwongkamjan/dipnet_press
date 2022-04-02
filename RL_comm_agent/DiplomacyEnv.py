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
    self.n_agents = 1
    self.sender_power = None
    self.stance = 0.0
    self.agent_id = [id for id in range(n_agents)]
    self.order_type_id = [id for id in range(5)]
    self.power_mapping = {}
    self.order_type_mapping = {}
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
    self.last_action_reward = 0

    
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
      done = {agent_id: True for agent_id in self.agent_id}
      reward = {agent_id: 0 for agent_id in self.agent_id}
      next_state = self.cur_obs # does not matter 
    else:  
      # censoring order - from deciding state to sent state (go to new state) or not send (stay at the same state)
      if action:
        done = {agent_id: False for agent_id in self.agent_id}
        reward = {agent_id: self.last_action_reward if agent_id == power_mapping[sender_power] else 0 for agent_id in self.agent_id} # get from next phase result (0/+1/-1 get/lose supply center)
        next_state = 
      else:
        done = {agent_id: False for agent_id in self.agent_id}
        reward = {agent_id: 0 for agent_id in self.agent_id}
        next_state = self.cur_obs
        
    return next_state, reward, done, {} #empty info
      
    

