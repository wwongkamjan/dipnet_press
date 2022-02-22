import asyncio
import random
from diplomacy.client.connection import connect
from diplomacy.utils import exceptions
import argparse

POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

class random_player:
  
  #get random orders
  def get_orders(self, game, power_name):
    print("random player get orders")
    if game.get_orderable_locations(power_name):
        possible_orders = game.get_all_possible_orders()
        orders = [random.choice(possible_orders[loc]) for loc in game.get_orderable_locations(power_name)
                  if possible_orders[loc]]
        return orders
      
  def get_message(self, game, msg_list, sender, recipient):
    return random.choice(msg_list)
