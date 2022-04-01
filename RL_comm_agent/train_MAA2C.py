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
stop_io_loop()           
if __name__ == '__main__':
  start_io_loop(main)
