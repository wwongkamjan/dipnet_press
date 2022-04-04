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
    self.state = 'no_order' # env state no_order -> censoring if censored -> no_order
                   #                                          if not -> share_order -> no_order 
    self.stance = 0.0
    self.agent_id = [id for id in range(n_agents)]
    self.order_type_id = [id for id in range(5)]
    self.power_mapping = {}
    self.order_type_mapping = {'move': 0, 'hold': 1, 'support':2, 'attack':3, 'convoy':4}
    """
    stance vector of [power1][power2],  
    send? 0/1
    orders:
            unit's power, England= ... one hot for 7 powers = [Eng, France, ..]
            type of order, one hot 5 [move, hold, support, attack, convoy]
            attack whom/move to whose territory one hot for 7 powers = [Eng, France, ..]
    cur_obs for each agent and action for each agent
    """
    self.observation_space = gym.spaces.Box(low=np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                                            high=np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
                                            dtype=np.float32)
    self.action_space = gym.spaces.Discrete(2) # 
    self.cur_obs = None
    self.episode_len = 0
    self.dip_net = None
    self.dip_game = None
    self.ep_states = []
    self.ep_actions = []
    self.ep_rewards = []
    self.ep_info = []
    self.ep_n_states = []
    self.ep_dones = []
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
    self.cur_obs = {agent_id: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for agent_id in self.agent_id} 
    return self.cur_obs
  
  def set_power_state(self, power_a, stance_of_power_b):
    self.cur_obs[self.power_mapping[power_a]][0] = stance_of_power_b
    
  def one_hot_order(self, order):
    order_token = get_order_tokens(order)
    if order_token[0][0] =='A' or order_token[0][0] =='F':
      # this is message about orders
      power1 = get_unit_power(order_token[0])
      if order_token[1] == 'S':
        order_type = 'support'
        order_unit = order_token[0][0]+' '+order_token[2]
        power2 =get_unit_power(order_unit)
        return one_hot(self.power_mapping[power1], self.n_agents) + one_hot(self.order_type_mapping[order_type],5) + one_hot(self.power_mapping[power2], self.n_agents)
        
      elif order_token[1] == 'H':
        order_type = 'hold'
        return one_hot(self.power_mapping[power1], self.n_agents) + one_hot(self.order_type_mapping[order_type],5) + [0.0]*self.n_agents
        
      elif order_token[1] == 'C':
        order_type = 'convoy'
        order_unit = order_token[0][0]+' '+order_token[2]
        power2 =get_unit_power(order_unit)
        return one_hot(self.power_mapping[power1], self.n_agents) + one_hot(self.order_type_mapping[order_type],5) + one_hot(self.power_mapping[power2], self.n_agents)
        
      else:
        #move/retreat or attack 
        #get location - add order_token[0] ('A' or 'F') at front to check if it collides with other powers' units
        order_unit = order_token[0][0]+' '+order_token[2]
        power2 =get_unit_power(order_unit)
        if power2:
          order_type= 'attack'
          return one_hot(self.power_mapping[power1], self.n_agents) + one_hot(self.order_type_mapping[order_type],5) + one_hot(self.power_mapping[power2], self.n_agents)
        else:
          order_type = 'move'
          return one_hot(self.power_mapping[power1], self.n_agents) + one_hot(self.order_type_mapping[order_type],5) + [0.0]*self.n_agents
        
  def get_unit_power(self, unit):
    for power in game.powers:
      if unit in game.powers[power].units:
        return power
      
  def one_hot(self, id, n):
    one_hot_list = [0.0 for i in range(n)]
    one_hot_list[id] = 1.0
    return one_hot_list
    
  def step(self, action, power_a, power_b, order): 
    """
    input:  action=dictionary of agent action where action = Discrete(2) or 0 or 1, 
            power_a = sender, 
            power_b = receiver, 
            order = order we're considering censor
    """

    
    if self.dip_game.game.is_game_done:
      done = {agent_id: True for agent_id in self.agent_id}
    else:  
      done = {agent_id: False for agent_id in self.agent_id}
      
    one_hot_order = self.one_hot_order(order)  
    self.ep_dones.append(done) 
    self.ep_actions.append(action)
    self.ep_states.append(self.cur_obs)
    self.ep_info.append((self.state, power_a, power_b, one_hot_order))
    
    if self.state =='no_order': 
      self.state == 'censoring'
      self.cur_obs[agent_id][2:] = one_hot_order 
      self.step(action, power_a, power_b, order)
      
    elif self.state == 'censoring':
      if action[power_a] ==0:
        self.state ='no_order'
        self.cur_obs[agent_id][2:] = [0.0]*19
      else:
        self.state = 'share_order'
        self.cur_obs[agent_id][1] = 1.0
        self.step(action, power_a, power_b, order)
    else:
      self.state = 'no_order'
      self.cur_obs[agent_id][1:] = [0.0]*20
 
  def get_transactions(self):
    #when the dip phase is done
    return  self.ep_states, self.ep_actions, self.ep_rewards, self.ep_n_states, self.ep_dones
    
  
  def diplomacy_step(self): move this to trainer
    self.phase_done = False
    for trainer copy/modify interact and train inside it 
    
    while not phase_done:
         action = self.action(state)
         env.take_action(action) 
          
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
              
            
    
      
    

