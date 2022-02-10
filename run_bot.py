from tornado import gen
import ujson as json
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
from diplomacy_research.players.benchmark_player import DipNetSLPlayer
from diplomacy_research.utils.cluster import start_io_loop, stop_io_loop
from diplomacy_research.models.state_space import get_order_tokens
from diplomacy.server.server_game import ServerGame
import random
import time

@gen.coroutine
def main():
    """ Plays a local game with 7 bots """
    player = DipNetSLPlayer()
    game = Game()
    game.remove_rule('NO_PRESS')
#     server_game = ServerGame(game)

    # Playing game
    while not game.is_game_done:
        orders = yield {power_name: player.get_orders(game, power_name) for power_name in game.powers}
        for power_name, power_orders in orders.items():
            # send out message randomly - to random powers
#             rec_power = random.choice(game.powers)
#             while rec_power == power_name or rec_power.is_eliminated():
#                 rec_power = random.choice(game.powers)
#             press_message = press(power_name, rec_power, power_orders)
            # send out every support order
            for order in power_orders:
                rec_power = None
                order_token = get_order_tokens(order)
                # find if an order is for self or other power
                if len(order_token) > 2:
                    for power2 in game.powers:
                        #check if the order's unit is in power unit list
                        if power2 != power_name and not game.powers[power2].is_eliminated() and order_token[2] in game.powers[power2].units:
                            rec_power = power2
                            
                # An order is to Support other power
                if order_token[1] =='S' and rec_power != None:
                    # send fact - support message
                    press_message = "press_msg from:"+power_name +" to:" + rec_power +" info: "+order
#                     print(game.powers[power_name].game.role)
#                     msg = game.new_power_message(power_name,rec_power, press_message)
                    msg = Message(sender=power_name,
                                  recipient=rec_power,
                                  message=press_message,
                                  phase=game.get_current_phase(),
                                  time_sent=int(time.time()))
                    game.add_message(msg)
            game.set_orders(power_name, power_orders)
        game.process()

    # Saving to disk
    with open('game.json', 'w') as file:
        file.write(json.dumps(to_saved_game_format(game)))
    stop_io_loop()

if __name__ == '__main__':
    start_io_loop(main)
