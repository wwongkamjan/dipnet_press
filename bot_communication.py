# script as helper for power-power communication
# class Dip Player get order, get message, reply
# class Diplomacy_Press:
# messages tracking sending/receiving/limitation/to_reply/generate all possible messages/replies
### generate all possible messages can be classified into groups (attacking) (supporting) (moving/holding/convoying) (proposals - (to attack) (to support) (to move/hold/convoy))
### generate all possible replies for non-proposal (okay) for proposal (yes,no,maybe)
# set_player - as DipNet/DORA/ random any thing

# main():
# set game, player for each power
# Player - get_orders/ get_messages/ get_replies
# Game - set_orders/ add_messages
# game.process()

Class Diplomacy_Press():
"""  Class for setting up players to play Diplomacy """
  def __init__(self, Game=None, Player=None, powers=None, number_msg_limitation=6):
    self.sent = {}
    self.received = {}
    self.player = Player()
    self.game = Game() 
    self.powers = powers
    self.number_msg_limitation = number_msg_limitation
    self.power_dict = {}
    for i in range(len(powers)):
      power_dict[powers[i]] = i
      
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
  
  def new_message(self, sender, recipient, message):
    self.game.add_message(message)
    
  def get_all_possible_message(self, sender, recipient):
    # include no message!

    possible_messages = []
    return possible_messages
  
  def get_all_possible_replies(self, sender, recipient):
    # include no message!
#     msg = Message(sender=sender,
#               recipient=recipient,
#               message=message,
#               phase=self.game.get_current_phase())
    possible_replies = []
    return possible_replies
  
#   def random_messages(self, possible_messages):
#     # we can set weights for groups of messages (later)
#     return random.choice(possible_messages)
  
  def send_message(self, sender, recipient):
#     if agent_type =="NO_PRESS":
#       # choose arbitrary messages (e.g. random, )
#       message = self.random_messages(self.get_all_possible_message(sender,recipient))
#     else:
    message = self.player.get_message(self.game, sender, recipient)

    msg = Message(sender=sender,
         recipient=recipient,
         message=message,
         phase=self.game.get_current_phase())
      
    if not self.sent[sender]:
      self.sent[sender] = {}
      
    self.sent[sender][recipient] = message
  
  def reply_messages(self, sender, recipient):
#     if agent_type =="NO_PRESS":
#       # choose arbitrary messages (e.g. random, )
#       message = self.random_messages(self.get_all_possible_replies(sender,recipient))
#     else:
    message = self.player.get_reply(self.game, sender, recipient)

    msg = Message(sender=sender,
         recipient=recipient,
         message=message,
         phase=self.game.get_current_phase())
      
    if not self.sent[sender]:
      self.sent[sender] = {}
      
    self.sent[sender][recipient] = message
    return None
#   def set_player(self, Player):
#     self.player = Player()
#   def set_game(self, Game):
#     self.game = Game() 
#   def set_powers(self, Game):
#     self.game = Game() 
  def get_orders(self):
    return {power_name: self.player.get_orders(self.game, power_name) for power_name in self.game.powers}
  def set_orders(self, power_name, power_orders):
    return self.game.set_orders(power_name, power_orders)
  def game_process(self):
    self.game.process()
    
    
    
  

