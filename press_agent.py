import asyncio
import random
from diplomacy.client.connection import connect
from diplomacy.utils import exceptions
import argparse

  
def get_message(game, stance, msg_list, sender, recipient): 
  # input: sender, recip, stance, message history, orders from DipNet 
  # output: message_list 
  
  # decide if want to send message 
  boolean_message_content = [True, False]
  for key in msg_list:
    if random.choice(boolean_message_content):
      # set message for this content type, e.g. pick sender moves 4 out of 10 orders - for now let's do select all - do nothing
      if key =='power_message' or key =='sender_proposal':
#         #randomly select one power that we want to share power's messages to recipient       
#         power_message = random.choice(list(msg_list['power_message'].values()))
#         # we can make sure that we wont set None which is possible from picking message from the received list but for now None is fine
#         msg_list['power_message'] = power_message  
          msg_list[key] = None # we care only sender_move for now
      else:
        # case when key = sender_move
        if stance[sender][recipient] == 'N':
          msg_list[key] = filter_message(game, msg_list['sender_move'], recipient, ['attack','hold','convoy'])
        elif stance[sender][recipient] == 'B':
           msg_list[key] = None
            
    else:
      msg_list[key] = None
  return msg_list

def filter_message(game, msg_list, power_name, type):
  #msg_list = list of string message
  # type - a message type to exclude from message_list e.g. ['attack', 'support', 'proposal', 'to_order', etc.]
  remove_list = []
  for msg in msg_list:
    if self.get_message_type(game, msg, power_name) in type:
      remove_list.append(msg)
    
return [msg for msg in msg_list if msg not in remove_list]

def get_proposal(game, sender, recipient):
  # propose all units move for recipient
  if game.get_orderable_locations(recipient):
      possible_orders = game.get_all_possible_orders()
      orders = [random.choice(possible_orders[loc]) for loc in game.get_orderable_locations(recipient)
                if possible_orders[loc]]
      return orders  
    
