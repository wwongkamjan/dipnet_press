import asyncio
import random
from diplomacy.client.connection import connect
from diplomacy.utils import exceptions
import argparse

POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

class random_player:
  
  #get random orders
  def get_orders(self, game, power_name):
#     print("random player get orders")
    if game.get_orderable_locations(power_name):
        possible_orders = game.get_all_possible_orders()
        orders = [random.choice(possible_orders[loc]) for loc in game.get_orderable_locations(power_name)
                  if possible_orders[loc]]
        return orders
      
  def get_message(self, game, msg_list, sender, recipient):
    # decide if want to send message 
    boolean_message_content = [True, False]
    for key in msg_list:
      if random.choice(boolean_message_content):
        # set message for this content type, e.g. pick sender moves 4 out of 10 orders - for now let's do select all - do nothing
        if key =='power_message':
          power_name, power_message = random.choice(msg_list['power_message'])
          msg_list['power_message'] = power_message
      else:
        msg_list[key] = None
    return msg_list
  
  def get_proposal(self, game, sender, recipient):
    # propose all units move for recipient
    if game.get_orderable_locations(recipient):
        possible_orders = game.get_all_possible_orders()
        orders = [random.choice(possible_orders[loc]) for loc in game.get_orderable_locations(recipient)
                  if possible_orders[loc]]
        return orders  
    
