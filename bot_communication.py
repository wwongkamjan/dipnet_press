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
from random_bot import random_player
from random_comm_functions import get_orders, get_proposal, get_message
from press_agent import get_message as press_get_message, get_proposal as press_get_proposal
import random
import time
import asyncio
import json
import csv
 

# script as helper for power-power communication
# class Dip Player get order, get message, reply
# class Diplomacy_Press:
# messages tracking sending/receiving/limitation/to_reply/generate all possible messages/replies
### generate all possible messages can be classified into groups (attacking) (supporting) (moving/holding/convoying) (proposals - (to attack) (to support) (to move/hold/convoy))
### generate all possible replies for non-proposal (okay) for proposal (yes,no,maybe) and counter proposal (move to A, move to B)
# set_player - as DipNet/DORA/ random any thing

# main():
# set game, player for each power
# Player - get_orders/ get_messages/ get_replies - communication is not allow in retreat and building phase (later)
# Game - set_orders/ add_messages
# game.process()
# agressive move = move to a map location that other powers' unit is currently on 

class Diplomacy_Press:
  def __init__(self, Game=None, Player=None, powers=None, number_msg_limitation=6):
    self.sent = {}
    self.received = {}
    self.proposal_sent = {}
    self.proposal_received = {}
    self.game = Game
    self.game.remove_rule('NO_PRESS')
    self.player = Player
    self.powers = self.game.powers
    self.number_msg_limitation = number_msg_limitation
    self.number_sent_msg = {}
#     self.power_dict = {}
#     for i in range(len(powers)):
#       power_dict[powers[i]] = i
            
    for power in self.powers:
      self.sent[power] = {power_name: None for power_name in self.powers}
      self.received[power] = {power_name: None for power_name in self.powers}
      self.proposal_sent[power] = {power_name: None for power_name in self.powers}
      self.proposal_received[power] = {power_name: None for power_name in self.powers}
      self.number_sent_msg[power] = 0
    
  def get_power_messages(self, power_name):
    return self.game.filter_messages(messages=self.game.messages, game_role=power_name)

  def get_every_sent_message(self, power_name):
    # any time during game
    messages=self.game.messages
    return {message.time_sent: message
        for message in messages
        if message.sender == power_name}

  def get_every_received_message(self, power_name):
    # any time during game
    messages=self.game.messages
    return {message.time_sent: message
        for message in messages
        if message.recipient == power_name}
  
  def get_received_message(self, power_name):
    # at current season
    return self.received[power_name]
  
  def get_sent_message(self, power_name):
    # at current season
    return self.sent[power_name]
  
  def new_message(self, DAIDE_message):
    self.game.add_message(DAIDE_message)
  
  @gen.coroutine 
  def get_all_possible_message(self, sender, recipient):
    # return dict_messages -> {'None' = None, 'sender_move': get_orders, 'sender_proposal': get_proposals (i.e. XDO request), '(other)power_message': get_received_message }
    # include no message!
    # at first, moves -> then proposal allies, enemies -> then XDO ...
    possible_messages = {}
    possible_messages['None'] = None
    # retrieve sender moves
    if self.game.get_current_phase() != 'S1901M':
     self.player.update_stance( self.game, sender) # update stance about other power before getting orders 
    orders = yield self.player.get_orders(self.game, sender)
#     orders = [ord for ord in order]

    possible_messages['sender_move'] = orders # will be later 'AND/OR'
    
    # # retrieve orders to propose to recipient
    # proposals = self.player.get_proposal(self.game, sender, recipient)
    # possible_messages['sender_proposal'] = proposals # will be later 'AND/OR'
    
    # # retrieve info of other power to forward/share to recipient
    # power_message = self.get_received_message(sender)
    # possible_messages['power_message'] = power_message # will be later 'AND/OR'
    return possible_messages
  
  @gen.coroutine 
  def get_all_possible_replies(self, sender, recipient):
    # include no reply, ignore the received message from this sender! 
    # include counter proposal
    possible_replies = ['None']
    possible_replies += ['Acknowledge','RejectProposal','AcceptProposal']
    return possible_replies
  
  @gen.coroutine
  def send_message(self, sender, recipient):
    # number of messages is not exceed limitation (e.g. 6 per phases) and the last message is replied by this recipient or never send to this recipient
    # if self.number_sent_msg[sender] <  self.number_msg_limitation and self.sent[sender][recipient]==None:
    msg_list = yield self.get_all_possible_message(sender, recipient)
    message  = yield self.player.get_message(self.game, msg_list, sender, recipient)
    if message:
      msg = Message(sender=sender,
            recipient=recipient,
            message=message,
            phase=self.game.get_current_phase())
            
      # sender sent message to recipient and recipient received message from sender
      self.sent[sender][recipient] = msg_list['sender_move']
      self.received[recipient][sender] = msg_list['sender_move']
      self.new_message(msg)
      # self.number_sent_msg[sender] += 1
#     else:
#         print("number of sent messages exceeds")
        
  @gen.coroutine 
  def reply_message(self, sender, recipient):
    # this is to reply a message, so sender becomes recipient and recipient becomes sender
    if self.received[sender][recipient]:
      msg_list = yield self.get_all_possible_replies(sender, recipient)
      message = self.player.get_reply(self.game, msg_list, sender, recipient)
      if message != "None":
        msg = Message(sender=sender,
             recipient=recipient,
             message=message,
             phase=self.game.get_current_phase())

        # set message to be None so that recipient can send a new message to a sender 
        self.sent[recipient][sender] = None
        self.received[sender][recipient] = None
        self.new_message(msg)
        self.number_sent_msg[recipient] -= 1
      
#     else:
#       raise "There is no message from " +recipient+ " to " +sender +" to reply"

  def get_orders(self):
    return {power_name: self.player.get_orders(self.game, power_name) for power_name in self.game.powers}

  def set_orders(self, power_name, power_orders):
    return self.game.set_orders(power_name, power_orders)

  def game_process(self):
    # reset contraints e.g. self.sent/received = None, self., self.number_sent_msg = 0
    for power in self.powers:
      self.sent[power] = {power_name: None for power_name in self.powers}
      self.received[power] = {power_name: None for power_name in self.powers}
      self.proposal_sent[power] = {power_name: None for power_name in self.powers}
      self.proposal_received[power] = {power_name: None for power_name in self.powers}
    self.game.process()

class Diplomacy_Press_Player:
 
  def __init__(self, bot_type=[], Player=None):
    self.player = Player
    self.stance = {}
    self.bot_temp = bot_type
    
  def init_communication(self, power_list):
    self.bot_type={}
    for power in power_list:
      self.stance[power] = {power: 0.0 for power in power_list}
      self.bot_type[power]= self.bot_temp.pop()
      # print('power: '+power+' as ', self.bot_type[power])
      
  def set_stance(self, power_name, status):
    self.stance[power_name] = status
    
  def update_stance(self, game, power_name): #use as basic to update stance of every other power for power_name 
    previous_order = game.get_orders()
    for other_power in self.stance:
      if power_name != other_power:
        if self.stance[power_name][other_power] < -1:
          self.stance[power_name][other_power] +=1
        for order in previous_order[other_power]:
          # print(order)
          if self.get_order_type(game, order, other_power,power_name) =='attack':
            self.stance[power_name][other_power] = -10.0
            # print('update stance [{}][{}]= {}'.format(power_name,other_power,self.stance[power_name][other_power]))
          elif self.get_order_type(game, order, other_power,power_name) =='support' and self.stance[power_name][other_power] >-1:
              self.stance[power_name][other_power] = 10.0
              # print('update stance [{}][{}]= {}'.format(power_name,other_power,self.stance[power_name][other_power]))
         
  @gen.coroutine 
  def get_orders(self, game , power_name):
    
#     await self.player.get_orders(game, power_name)
    # print('get_orders')
    future_or_iterable = yield self.player.get_orders(game, power_name)
    # print(future_or_iterable)
    orders = [future_or_iterable] if not isinstance(future_or_iterable, list) else future_or_iterable
    # print("DPP.get_orders()",orders)
    # while [1 for order in orders if not order.done()]:
    #  print('in while')
    # orders = yield [order.result() for order in orders]
    # print('orders', orders)
    return orders
 
  @gen.coroutine 
  def get_message(self, game, msg_list, sender, recipient):
    # if agent is no press, you can call random/non-attacking messages we provided i.e. self.random_message_list(msg_list)
    # else call you agent to send message from sender to recipient
    #return string of message
    
    #filter out agressive message from sender_move i.e. attacking message
    
    # let agent choose message for each category 
    # if random player
#     msg_list = self.player.get_message(game, msg_list, sender, recipient)

    # if dipnet
#     msg_list = get_message(game, msg_list, sender, recipient) # random select 
    # if self.bot_type[sender] =='transparent':
    #   msg_list = get_message(game, msg_list, sender, recipient)
    # elif self.bot_type[sender] =='press_agent':
    #   msg_list = press_get_message(game, self.stance, msg_list, sender, recipient) # strategically select
    
    # join string for sender move
    # AND (FCT (order1)) ((FCT (order2))) ..
    message_str = ''
    if msg_list['sender_move']:
      # msg_list['sender_move'] = self.filter_message(game, msg_list['sender_move'], sender, ['attack']) #censor aggressiv move
      sender_move_str = [' ( FCT ( '+order+' ) )' for order in msg_list['sender_move']]
      sender_move_str = ''.join(sender_move_str)
      message_str += 'AND '+ sender_move_str
    message_str = 'stance['+sender+']['+recipient +']=' +str(self.stance[sender][recipient]) + message_str
    # join string for proposal
    # AND (PRP (order1)) ((FCT (order2))) ..
    # if msg_list['sender_proposal']:
    #   # msg_list['sender_proposal'] = self.filter_message(game, msg_list['sender_proposal'], recipient, ['attack']) #censor aggressiv movezzz
    #   sender_proposal_str = [' ( XDO ( '+order+' ) )' for order in msg_list['sender_proposal']]
    #   sender_proposal_str = ''.join(sender_proposal_str)
    #   message_str += ' power_proposal: ' +sender_proposal_str
    
    # # message from other power that you want to share (agent already select specific power)
    # if msg_list['power_message']:
    #   # expect (1) sender move (2) proposal (3) moves of other power, for now sharing only (1)
    #   message_split = msg_list['power_message'].split(':') 
    #   if message_split[0] == 'power_move':
    #    message_str += ' other_info: ' +message_split[1]
    
    if len(message_str)==0:
      return None
    else:
      return message_str

  def get_reply(self, game, msg_list, sender, recipient):
    return self.random_message_list(msg_list)
  
  def get_proposal(self, game, sender, recipient):
    # what moves to propose to recipient?
    #if dipnet
    if self.bot_type[sender] =='transparent':
      return get_proposal(game, sender, recipient)
    elif self.bot_type[sender] =='press_agent':
      return press_get_proposal(game, sender, recipient)
    
    #if random player
#     return self.player.get_proposal(game, sender, recipient)

  def random_message_list(self, msg_list):
    return random.choice(msg_list)
  
  def filter_message(self, game, msg_list, power_name, type):
    #msg_list = list of string message
    # type - a message type to exclude from message_list e.g. ['attack', 'support', 'proposal', 'to_order', etc.]
    remove_list = []
    for msg in msg_list:
      if self.get_message_type(game, msg, power_name) in type:
        remove_list.append(msg)
    
    return [msg for msg in msg_list if msg not in remove_list]
  
  def get_message_type(self, game, msg, power_name):
    # check if it is support?
    # attack?
    # move? 
    # convoy
    # hold
    order_token = get_order_tokens(msg)
    if order_token[0] =='A' or order_token[0] =='F':
      # this is message about orders
      if order_token[1] == 'S':
        return 'support'
      elif order_token[1] == 'H':
        return 'hold'
      elif order_token[1] == 'C':
        return 'convoy'
      else:
        #move/retreat or attack 
        #get location - add order_token[0] ('A' or 'F') at front to check if it collides with other powers' units
        order_unit = order_token[0]+' '+order_token[2]
        #check if loc has some units of other powers on
        for power in game.powers:
          if power_name != power:
            if order_unit in game.powers[power].units:
              return 'attack'
            else:
              return 'move'  
             
  def get_order_type(self, game, order, power_sub, power_obj):
    order_token = get_order_tokens(order)
    # print('order:')
    # print(order_token)
    if order_token[0][0] =='A' or order_token[0][0] =='F':
      # this is message about orders
      if order_token[1] == 'S':
        order_unit = order_token[2]
        if order_unit in game.powers[power_obj].units:
          return 'support'
      elif order_token[1] == 'H':
        return 'hold'
      elif order_token[1] == 'C':
        return 'convoy'
      else:
        #move/retreat or attack 
        #get location - add order_token[0] ('A' or 'F') at front to check if it collides with other powers' units
        order_unit = order_token[0][0] + order_token[1][1:]
        if order_unit in game.powers[power_obj].units:
          return 'attack'
        else:
          return 'move'  

 
@gen.coroutine
def main():
  bots = ['transparent','transparent','transparent','transparent','transparent','transparent','press_agent']
  dip_player =  Diplomacy_Press_Player(bots, Player=DipNetSLPlayer())
#   dip_player =  Diplomacy_Press_Player(Player=random_player())
  dip_game =  Diplomacy_Press(Game=Game(), Player=dip_player)
  dip_player.init_communication(dip_game.game.powers)
  while not dip_game.game.is_game_done:
    print(dip_game.game.get_current_phase())
    if dip_game.game.phase_type != 'A' and dip_game.game.phase_type != 'R': # no communication during retreat and building phase
      #send messages before taking orders
      for sender in dip_game.powers:
        for recipient in dip_game.powers:
          if sender != recipient and not dip_game.powers[sender].is_eliminated() and not dip_game.powers[recipient].is_eliminated() :
            yield dip_game.send_message(sender, recipient)
      #reply to messages - game/allies/enemy state (or stance) can be changed after getting messages and replies
      for sender in dip_game.powers:
        for recipient in dip_game.powers: 
          if sender != recipient and not dip_game.powers[sender].is_eliminated() and not dip_game.powers[recipient].is_eliminated():
            yield dip_game.reply_message(sender, recipient)

    #taking orders after messages were all sent
    orders = yield {power_name: dip_player.get_orders(dip_game.game, power_name) for power_name in dip_game.powers}
#     orders = {power_name: dip_player.get_orders(dip_game.game, power_name) for power_name in dip_game.powers}
    for power_name, power_orders in orders.items():
       dip_game.game.set_orders(power_name, power_orders)
    dip_game.game_process()
     # Saving to disk
  game_history_name = '1vs6_press_agent_vs_transparent'
  exp = game_history_name
  game_history_name += '.json'
  with open(game_history_name, 'w') as file:
    file.write(json.dumps(to_saved_game_format(dip_game.game)))
   
  # Opening JSON file and loading the data
  # into the variable data
  with open(exp + '.json') as json_file:
    data = json.load(json_file)
  game_data = data['phases']
  data_file = open(exp + '.csv', 'w')
  
  csv_writer = csv.writer(data_file)
  count = 0
  for phase in game_data:
    if count == 0:
        # Writing headers of CSV file
        header = phase.keys()
        csv_writer.writerow(header)
        count += 1

    # Writing data of CSV file
    csv_writer.writerow(phase.values())

  data_file.close()
  stop_io_loop()

if __name__ == '__main__':
  start_io_loop(main)

