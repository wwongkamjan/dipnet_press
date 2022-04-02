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
    self.sending = True
    self.stance = 0.0
    self.agent_id = [id for id in range(n_agents)]
    self.order_type_id = [id for id in range(5)]
    self.power_mapping = {}
    self.order_type_mapping = {'move': 0, 'hold': 1, 'support':2, 'attack':3, 'convoy':4}
    """
    stance vector of [power1][power2],  
    orders:
            unit's power, England= ... one hot for 7 powers = [Eng, France, ..]
            type of order, one hot [move, hold, support, attack, convoy]
            attack whom/move to whose territory one hot for 7 powers = [Eng, France, ..]
    cur_obs for each agent and action for each agent
    """
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
    self.dip_player =  Diplomacy_Press_Player(Player=DipNetSLPlayer())
    self.dip_game =  Diplomacy_Press()
    self.dip_player.init_communication(self.dip_game.powers)
    self.power_mapping = {power: id for power,id in zip(self.dip_game.powers,self.agent_id)}
    self.episode_len = 0
    # initial state = neutral for any power and no order OR having not assigned sender, recipient yet
    self.cur_obs = {agent_id: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for agent_id in self.agent_id} 
    return self.cur_obs
    
  def step(self, action): 
    """
    input: Discrete(2) - 0 or 1
    output: return state, reward, done, info
    """
    if self.dip_game.game.is_game_done:0
      done = {agent_id: True for agent_id in self.agent_id}
      reward = {agent_id: 0 for agent_id in self.agent_id}
      next_state = self.cur_obs # does not matter 
    else:  
      # censoring order - from deciding state to sent state (go to new state) or not send (stay at the same state)
      done = {agent_id: False for agent_id in self.agent_id}
      if self.sending: 
        if action:
          # reward will be from next phase result (0/+1/-1 get/lose supply center)
          reward = {agent_id: self.last_action_reward if agent_id == self.power_mapping[sender_power] else 0 for agent_id in self.agent_id} 
          next_state = {agent_id: [self.stance, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] if agent_id ==self.power_mapping[sender_power] else 0 for agent_id in self.agent_id} 
        else:
          reward = {agent_id: 0 for agent_id in self.agent_id}
          next_state = self.cur_obs   
        self.sending = False
      else:
        # the order is sent - taking any action to go back to initial state 
        reward = {agent_id: 0 for agent_id in self.agent_id}
        next_state= {agent_id: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for agent_id in self.agent_id}
        self.sending = True
        
    return next_state, reward, done, {} #empty info
  
  def diplomacy_step(self):
    self.phase_done = False
    for Dip MAA2C agent you can 
    while not phase_done:
         env.step(action)
    states, actions, rewards, next_states, dones = env.get_transactions()
    for s,a,r,n_s,d zip(states, actions, rewards, next_states, dones)
      if done:
        r = discount similarly to org
      else: 
        r = discount
      memory.push(s,a,r)
    for sender in self.dip_game.powers:
        for recipient in self.dip_game.powers:
          if sender != recipient and not dip_game.powers[sender].is_eliminated() and not dip_game.powers[recipient].is_eliminated() :
            take multiple steps i.e. let the power iterate through k first orders from DipNet, censor them and then send message
            draft:
              1. set self.sender_power = sender
              2. get_orders from DipNet, pick best k
              3. for each order:
                  - set curr_obs (now it is all zeros state) to be [dip_player.stance[sender][power], zeros] (now stance is A (+10), B (-10), N (0)) 
                  - in step() add state = cur_obs to the list and add next_state_info as tuple (sender, recipinet, one hot order)
              4. send message about selected orders
              5. recipient acknowledge 
              6. power do actions and game.process
              7. get stance[sender][recipient], reward from new phase 
              8. tranform tuples in next_state_info to [dip_player.stance[sender][recipient], one-hot order (line26-30)] and add to the next_states list
              9. add reward, done to the list - how to reward? let's do DipNet? or relavance of taken orders and shared orders?
   self.phase_done = True
              
            
    
      
    

