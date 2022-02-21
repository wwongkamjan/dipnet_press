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

@gen.coroutine
def main():
    """ Plays a local game with 7 bots """
    player = DipNetSLPlayer()
    game = Game()
    game.remove_rule('NO_PRESS')
    req = RequestBuilder()
#     server_game = ServerGame(game)

    # Playing game
    while not game.is_game_done:
        orders = yield {power_name: player.get_orders(game, power_name) for power_name in game.powers}
        for power_name, power_orders in orders.items():
            # send out every non-attacking order
            for order in power_orders:
                attack = False
                rec_power = None
                order_token = get_order_tokens(order)
                # find if an order is for self or other power
#                 if len(order_token) > 2:
#                     # check for support msg
#                     for power2 in game.powers:
#                         #check if the order's unit is in power unit list
#                         
#                         if power2 != power_name and not game.powers[power2].is_eliminated() and order_token[2] in game.powers[power2].units:
#                             rec_power = power2
                            
                # filter for non-attacking orders
                order_token2 = order_token[1].split() 
                for power2 in game.powers:
                    if len(order_token2) > 1:
                        if power2 != power_name and not game.powers[power2].is_eliminated() and order_token2[1] in game.powers[power2].units:
                            attack=True
                # skip attacking order
                if attack:
                    continue
                             
#               send non-attacking message / move, hold, (self-)support, convoy - randomly to other powers with prob e
                e = 0.3
                if e >= random.uniform(0, 1):
                    rec_power = random.choice(list(game.powers.keys()))
                    while rec_power == power_name or game.powers[rec_power].is_eliminated():
                        rec_power = random.choice(list(game.powers.keys()))
                    press_message = "SND ( "+power_name+" ) ( "+rec_power+" ) FCT ( "+order+" )" 
                    msg = Message(sender=power_name,
                                  recipient=rec_power,
                                  message=press_message,
                                  phase=game.get_current_phase())
                    game.add_message(msg)
#                     msg_byte = bytes(press_message, encoding='utf8')
#                     print(msg_byte[:3])
#                     daide_req = req.from_bytes(msg_byte)
#                     print(daide_req.__str__())   
            game.set_orders(power_name, power_orders)
        message_in_phase = game.filter_messages(messages=game.messages, game_role="FRANCE")
        print(game.get_current_phase())
        for m in message_in_phase.values():
            print(m)
        game.process()

    # Saving to disk
    with open('game.json', 'w') as file:
        file.write(json.dumps(to_saved_game_format(game)))
    stop_io_loop()

if __name__ == '__main__':
    start_io_loop(main)
