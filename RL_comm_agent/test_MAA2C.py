from pytorch_DRL.MAA2C import MAA2C
from pytorch_DRL.common.utils import ma_agg_double_list, dict_to_arr, arr_dict_to_arr, index_to_one_hot
from tornado import gen
import ujson as json
import csv

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import random

# MAA2C: https://github.com/ChenglongChen/pytorch-DRL/
from DiplomacyEnv import DiplomacyEnv
from diplomacy.engine.message import Message
from diplomacy.utils.export import to_saved_game_format
from diplomacy_research.utils.cluster import start_io_loop, stop_io_loop

MAX_EPISODES = 10
EPISODES_BEFORE_TRAIN = 2
TEST_EPISODES = 10
EVAL_INTERVAL = 2

# roll out n steps
ROLL_OUT_N_STEPS = 20
# only remember the latest 2 ROLL_OUT_N_STEPS
MEMORY_CAPACITY = 10*ROLL_OUT_N_STEPS
# only use the latest 2 ROLL_OUT_N_STEPS for training A2C
BATCH_SIZE = 2*ROLL_OUT_N_STEPS

REWARD_DISCOUNTED_GAMMA = 0.99
ENTROPY_REG = 0.00

DONE_PENALTY = 0

CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

RANDOM_SEED = 2017
N_AGENTS = 7
K_ORDERS = 5
AGENT = None
EVAL_REWARDS = None

@gen.coroutine
def test():
    global EVAL_REWARDS
    global AGENT
    hist_name = 'comm_agent'
    env = DiplomacyEnv()
    rewards = []
    
    state_dim = env.observation_space.shape[0]
    if len(env.action_space.shape) > 1:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    maa2c = MAA2C(env=env, n_agents=N_AGENTS, 
            state_dim=state_dim, action_dim=action_dim, memory_capacity=MEMORY_CAPACITY,
            batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
            done_penalty=DONE_PENALTY, roll_out_n_steps=ROLL_OUT_N_STEPS,
            reward_gamma=REWARD_DISCOUNTED_GAMMA,
            epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
            episodes_before_train=EPISODES_BEFORE_TRAIN, training_strategy="centralized",
            critic_loss=CRITIC_LOSS, actor_parameter_sharing=True, critic_parameter_sharing=True)  

    maa2c.load_model('models/a2c_actor_diplomacy_ep_9_v1', 'models/a2c_critic_diplomacy_ep_9_v1')

    dip_step = 0

    dip_game = env.dip_game
    dip_player = env.dip_player
    bot_type = ['RL', 'RL', 'transparent', 'transparent', 'transparent', 'transparent', 'transparent']
    random.shuffle(bot_type)
    dip_player.bot_type = {power: b for b,power in zip(bot_type, dip_game.powers)}
    id = 0

    while not dip_game.game.is_game_done:
        if dip_game.game.phase_type != 'A' and dip_game.game.phase_type != 'R':
            centers = {power: len(dip_game.game.get_centers(power)) for power in dip_game.powers}
            for sender in dip_game.powers:
                for recipient in dip_game.powers:
                    share_order_list = []
                    if sender != recipient and not dip_game.powers[sender].is_eliminated() and not dip_game.powers[recipient].is_eliminated():
                        if dip_player.bot_type[sender] == 'transparent':
                            yield dip_game.send_message(sender, recipient)
                        elif dip_player.bot_type[sender] == 'RL':
                            orders = yield dip_player.get_orders(dip_game.game, sender)
                            stance = dip_player.stance[sender][recipient] 
                            n = len(orders)
                            env.set_power_state(sender, stance)
                            # print('sender: ', sender + ' recipient: ', recipient)
                            for order in orders[:min(K_ORDERS,n)]:
                                # print('consider: ', order)
                                maa2c.env_state = dict_to_arr(env.cur_obs, N_AGENTS)
                                action = maa2c.exploration_action(maa2c.env_state)
                                action_dict = {agent_id: action[agent_id] for agent_id in range(maa2c.n_agents)}
                                env.step(action_dict, sender, recipient, order)
                                # if action=share, we add it to the list
                                if action_dict[env.power_mapping[sender]]==1:
                                    share_order_list.append(order)

                            dip_game.received[recipient][sender] = share_order_list

                            env.reset_power_state(sender, recipient)
                            
                            message = [' ( FCT ( '+order+' ) )' for order in share_order_list]
                            message = ''.join(message)
                            if len(message):
                                message = 'AND' + message
                            message = 'stance['+sender+']['+recipient +']=' +str(stance) + message
                            msg = Message(sender=sender,
                                        recipient=recipient,
                                        message=message,
                                        phase=dip_game.game.get_current_phase())
                            dip_game.new_message(msg)

        #generating an imagined world from received messages
        # new_orders = yield {power_name: orders_of_generated_game(dip_game.game, dip_player, power) for power_name in dip_game.powers}
        orders = yield {power_name: dip_player.get_orders(dip_game.game, power_name) for power_name in dip_game.powers}
        
        # print('new_orders: ', new_orders)
        # print('orders: ', orders)

        for power_name, power_orders in orders.items():
            dip_game.game.set_orders(power_name, power_orders)

        for power in dip_game.powers:
            dip_player.update_stance(dip_game.game, power)

        dip_game.game_process()

    centers_power = {power: len(dip_game.game.get_centers(power)) for power in dip_game.powers}
    print('evaluation result: ' )
    for power in dip_game.powers:
        print('%s: %d centers' % (power, centers_power[power]))
    
    # maa2c.save_model('diplomacy', 'ep_{}_v1'.format(str(maa2c.n_episodes)))
    save_to_json(hist_name, dip_game, dip_player.bot_type)
    EVAL_REWARDS = rewards
    stop_io_loop()


def save_to_json(name, game, bot_type):
    game_history_name = name + '_with_baseline_bots_2RLvs5Transparent' 
    exp = game_history_name
    game_history_name += '.json'
    with open(game_history_name, 'w') as file:
        file.write(json.dumps(to_saved_game_format(game.game)))
    
    # Opening JSON file and loading the data
    # into the variable data
    with open(exp + '.json') as json_file:
        data = json.load(json_file)
    game_data = data['phases']
    data_file = open(exp + '.csv', 'w')
    
    csv_writer = csv.writer(data_file)
    csv_writer.writerow(bot_type.keys())
    csv_writer.writerow(bot_type.values())
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
        
# @gen.coroutine
def main():    
    for i in range(TEST_EPISODES):
        start_io_loop(test)
  
        
if __name__ == '__main__':
    main()
