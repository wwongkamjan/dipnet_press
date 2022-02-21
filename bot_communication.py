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
import random
import time

# script as helper for power-power communication
# class Dip Player get order, get message, reply
# class Diplomacy_Press:
# messages tracking sending/receiving/limitation/to_reply/generate all possible messages/replies
### generate all possible messages can be classified into groups (attacking) (supporting) (moving/holding/convoying) (proposals - (to attack) (to support) (to move/hold/convoy))
### generate all possible replies for non-proposal (okay) for proposal (yes,no,maybe) and counter proposal (move to A, move to B)
# set_player - as DipNet/DORA/ random any thing

# main():
# set game, player for each power
# Player - get_orders/ get_messages/ get_replies
# Game - set_orders/ add_messages
# game.process()

class Diplomacy_Press:
  def __init__(self, Game=None, Player=None, powers=None, number_msg_limitation=6):
    self.sent = {}
    self.received = {}
    self.game = Game() 
    self.game.remove_rule('NO_PRESS')
    self.player = Player()
    self.powers = self.game.powers
    self.number_msg_limitation = number_msg_limitation
    self.number_sent_msg = {}
#     self.power_dict = {}
#     for i in range(len(powers)):
#       power_dict[powers[i]] = i
            
    for power in self.powers:
      self.sent[power] = {power_name: None for power_name in self.powers}
      self.received[power] = {power_name: None for power_name in self.powers}
      self.number_sent_msg[power] = 0
      
  def get_power_messages(self, power_name):
    return self.game.filter_messages(messages=self.game.messages, game_role=power_name)
  
  def get_sent_message(self, power_name):
    return {message.time_sent: message
        for message in messages
        if message.sender == power_name}
  
  def get_recieved_message(self, power_name):
    return {message.time_sent: message
        for message in messages
        if message.recipient == power_name}
  
  def new_message(self, DAIDE_message):
    self.game.add_message(DAIDE_message)
    
  def get_all_possible_message(self, sender, recipient):
    # include no message!
    # at first, moves -> then proposal allies, enemies -> then XDO ...
    possible_messages = ['None']
    possible_messages.append(self.player.get_orders(self.game, sender).join(' AND ')) #get_non-attacking_orders
    return possible_messages
  
  def get_all_possible_replies(self, sender, recipient):
    # include no reply, ignore the recieved message from this sender! 
    # include counter proposal
    possible_replies = ['None']
    possible_messages += ['Okay']
    return possible_replies

  def send_message(self, sender, recipient):
    # number of messages is not exceed limitation (e.g. 6 per phases) and the last message is replied by this recipient or never send to this recipient
    if self.number_sent_msg[sender] <  self.number_msg_limitation and self.sent[sender][recipient]==None:
      msg_list = get_all_possible_message(sender, recipient)
      message = self.player.get_message(self.game, msg_list, sender, recipient)
      if message != "None":
        msg = Message(sender=sender,
             recipient=recipient,
             message=message,
             phase=self.game.get_current_phase())
              
        # sender sent message to recipient and recipient recieved message from sender
        self.sent[sender][recipient] = message
        self.recieved[recipient][sender] = message
        self.new_message(msg)
        self.number_sent_msg[sender] += 1
      else:
        print("number of sent messages exceeds")
  
  def reply_messages(self, sender, recipient):
    # this is to reply a message, so sender becomes recipient and recipient becomes sender
    if self.recieved[sender][recipient]:
      msg_list = get_all_possible_replies(sender, recipient)
      message = self.player.get_reply(self.game, msg_list, sender, recipient)

      msg = Message(sender=sender,
           recipient=recipient,
           message=message,
           phase=self.game.get_current_phase())
                          
      # set message to be None so that recipient can send a new message to a sender 
      self.sent[recipient][sender] = None
      self.recieved[sender][recipient] = None
      self.new_message(msg)
      self.number_sent_msg[recipient] -= 1
      
#     else:
#       raise "There is no message from " +recipient+ " to " +sender +" to reply"
      
  def get_orders(self):
    return {power_name: self.player.get_orders(self.game, power_name) for power_name in self.game.powers}
  def set_orders(self, power_name, power_orders):
    return self.game.set_orders(power_name, power_orders)
  def game_process(self):
    self.game.process()
    
class Diplomacy_Press_Player:
  def __init__(self, Player=None):
    self.player = Player()
    
  def get_orders(self, game , power_name):
    return self.player.get_orders(game, power_name)
  
  def get_message(self, game, msg_list, sender, recipient):
    # if agent is no press, you can call random/non-attacking messages we provided
    # else call you agent to send message from sender to recipient
    #return string of message
    
    return self.random_message_list(msg_list)
  
  def get_reply(self, game, msg_list, sender, recipient):
    return self.random_message_list(msg_list)
  
  def random_message_list(self, msg_list):
    return random.choice(msg_list)
  
@gen.coroutine
def main():
  dip_player =  Diplomacy_Press_Player(Player=DipNetSLPlayer)
  dip_game =  Diplomacy_Press(Game=Game, Player=dip_player)
  while not dip_game.game.is_game_done:
    #send messages before taking orders
    for sender in dip_game.powers:
      for recipient in dip_game.powers:
        if sender != recipient:
          dip_game.send_message(self, sender, recipient)
    #reply to messages - game/allies/enemy state (or stance) can be changed after getting messages and replies
    for sender in dip_game.powers:
      for recipient in dip_game.powers:
        if sender != recipient:
          dip_game.reply_message(self, sender, recipient)
                             
    #taking orders after messages were all sent
    orders = yield {power_name: dip_player.get_orders(dip_game.game, power_name) for power_name in dip_game.powers}
    for power_name, power_orders in orders.items():
       dip_game.game.set_orders(power_name, power_orders)
    dip_game.process()
     # Saving to disk
  with open('game.json', 'w') as file:
    file.write(json.dumps(to_saved_game_format(game)))
  stop_io_loop()

if __name__ == '__main__':
    start_io_loop(main)

