# script as helper for power-power communication
# class Dip Player:
# messages tracking sending/receiving/limitation/to_reply/generate all possible messages/replies
### generate all possible messages can be classified into groups (attacking) (supporting) (moving/holding/convoying) (proposals - (to attack) (to support) (to move/hold/convoy))
### generate all possible replies for non-proposal (okay) for proposal (yes,no,maybe)
# set_player - as DipNet/DORA/ random any thing

# main():
# set game, player for each power
# Player - get_orders/ get_messages/ get_replies
# Game - set_orders/ add_messages
# game.process()

Class Diplomacy_game_player():
  """ Class for setting up players to play Diplomacy """
  def __init__(self, Game=None, Player=None, powers=None, number_msg_limitation=6):
    self.is_sent = {}
    self.is_received = {}
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
  def new_message(self, sender, receiver, message):
    msg = Message(sender=sender,
              recipient=receiver,
              message=message,
              phase=self.game.get_current_phase())
    self.game.add_message(msg)
    if not self.is_sent[sender]:
      self.is_sent[sender] = {}
    if not self.is_recieved[receiver]:
      self.is_recieved[receiver] = {}
    self.is_sent[sender][receiver] = True
    self.is_recieved[receiver][sender] = True
    
  def get_to_send(self, type):
    return None
  def get_to_reply(self):
    return None
  def send_messages(self):
    return None
  def reply_messages(self):
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
    
    
    
  

