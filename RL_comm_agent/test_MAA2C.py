from pytorch_DRL.MAA2C import MAA2C
from pytorch_DRL.common.utils import ma_agg_double_list, dict_to_arr, arr_dict_to_arr, index_to_one_hot
from tornado import gen
import ujson as json
import csv
import random

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
EPISODE = 0
EVAL_INTERVAL = 2
DISCOUNT_ALLY_REWARD = 0.7
DISCOUNT_ORDER_REWARD = 0.3

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
AGENT_VERSION = "v2" 

RANDOM_SEED = 1000
N_AGENTS = 7
K_ORDERS = 5
AGENT = None
EVAL_REWARDS = None

@gen.coroutine
def test():
    global EVAL_REWARDS
    global AGENT
    global EPISODE
    hist_name = 'comm_agent_{}'.format(AGENT_VERSION)
    env = DiplomacyEnv()
    rewards = []
    stance_rewards = []
    proposal_stat = []
    random.seed(RANDOM_SEED+random.randint(0, 999))
    
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

    maa2c.load_model('models/a2c_actor_diplomacy_proposal_{}'.format(AGENT_VERSION), 'models/a2c_critic_diplomacy_proposal_{}'.format(AGENT_VERSION))

    dip_step = 0

    dip_game = env.dip_game
    dip_player = env.dip_player
    bot_type = ['RL', 'transparent', 'transparent', 'transparent', 'transparent', 'transparent', 'transparent']
    random.shuffle(bot_type)
    dip_player.bot_type = {power: b for b,power in zip(bot_type, dip_game.powers)}
    id = 0
    # order_game_memo= {}
    last_ep_index = 0
    # RLagent_id = [env.power_mapping[power] for power, bot_type in dip_player.bot_type.items() if bot_type =='RL' ]
    RL_power = [power for power, bot_type in dip_player.bot_type.items() if bot_type =='RL' ]
    dict_stat = []
    while not dip_game.game.is_game_done:
        orders = yield {power_name: dip_player.get_orders(dip_game.game, power_name) for power_name in dip_game.powers}
        if dip_game.game.phase_type != 'A' and dip_game.game.phase_type != 'R':
            centers = {power: len(dip_game.game.get_centers(power)) for power in dip_game.powers}
            for power in dip_game.powers:
                ally_power = ''
                enemy_power = ''
                ally = 0
                enemy = 0
                for power2 in dip_game.powers:
                    if power != power2:
                        if dip_player.stance[power][power2] > 1:
                            ally += centers[power2]
                            ally_power += power2 + ' '
                        elif dip_player.stance[power][power2] < -1:
                            enemy += centers[power2]
                            enemy_power += power2 + ' '
                stance_rewards.append([dip_step, dip_player.bot_type[power], power, centers[power], ally_power, ally, enemy_power, enemy])

            for sender in dip_game.powers:
                for recipient in dip_game.powers:
                    share_order_list = []
                    if sender != recipient and not dip_game.powers[sender].is_eliminated() and not dip_game.powers[recipient].is_eliminated():
                        if dip_player.bot_type[sender] == 'transparent':
                            yield dip_game.send_message(sender, recipient)
                        elif power in RL_power:
                            power_orders = orders[power]
                            stance = dip_player.stance[sender][recipient] 
                            n = len(orders)
                            env.set_power_state(sender, stance)
                            # print('sender: ', sender + ' recipient: ', recipient)
                            for order in power_orders[:min(K_ORDERS,n)]:
                                # print('consider: ', order)
                                maa2c.env_state = dict_to_arr(env.cur_obs, N_AGENTS)
                                action = maa2c.exploration_action(maa2c.env_state)
                                action_dict = {agent_id: action[agent_id] for agent_id in range(maa2c.n_agents)}
                                env.step(action_dict, sender, recipient, order)
                                # if action=share, we add it to the list
                                if action_dict[env.power_mapping[sender]]==1:
                                    share_order_list.append(order)
                                order_info = env.translate_order(order)
                                dict_stat.append([sender, recipient, action_dict[env.power_mapping[sender]], order_info[0],order_info[1],order_info[2] if len(order_info)>2 else None, env.get_power_type(sender,recipient)])

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
            if AGENT_VERSION == 'v2':
                for power in dip_game.powers:
                    if dip_player.bot_type[power] =='RL':
                        orders[power] = yield orders_of_generated_game(dip_game, dip_player, power)

            # proposal
            for sender in dip_game.powers:
                for recipient in dip_game.powers:
                    stance = dip_player.stance[sender][recipient] 
                    proposal_order_list = []
                    if dip_player.bot_type[sender] == 'RL':
                    # scan through list of dipnet order
                        rep_orders = orders[recipient] 
                        for order in rep_orders:
                            order_info = env.translate_order(order)
                            bool_propose = False
                            # if they have order that supporting sender, propose this
                            if len(order_info)>2:
                                if order_info[1] == 'support' and order_info[2]==sender:
                                    bool_propose = True
                                # if they have order that attacking enemy of sender, propose this
                                if order_info[1] == 'attack' and env.get_power_type(sender,order_info[2]) =='enemy':
                                    bool_propose = True
                                if bool_propose:
                                    proposal_order_list.append(order)
                        # print('RL')
                    if dip_player.bot_type[sender] == 'transparent':
                        proposal_order_list = dip_player.get_proposal(dip_game.game, sender, recipient) if dip_player.get_proposal(dip_game.game, sender, recipient) else []
                        # print('tran')
                    if len(proposal_order_list):
                        dip_game.proposal_received[recipient][sender] = proposal_order_list
                        message = [' ( PRP ( '+order+' ) )' for order in proposal_order_list]
                        message = ''.join(message)
                        if len(message):
                            message = 'AND' + message
                        message = 'stance['+sender+']['+recipient +']=' +str(stance) + message
                        msg = Message(sender=sender,
                                    recipient=recipient,
                                    message=message,
                                    phase=dip_game.game.get_current_phase())
                        dip_game.new_message(msg)
                        # print('proposal sent')   
                           
            # proposal process
            #if not enemy, I will follow your order + my own
           
            
            for recipient in dip_game.powers:
                for sender in dip_game.powers:
                    stance = dip_player.stance[recipient][sender] 
                    if dip_game.proposal_received[recipient][sender]:
                        answer = 'REJ'
                        if env.get_power_type(recipient,sender)=='ally':
                            answer = 'YES'
                        
                        message = [' ( {} ( '.format(answer) +order+' ) )'for order in dip_game.proposal_received[recipient][sender]]
                        message = ''.join(message)
                        if len(message):
                            message = 'AND' + message
                        message = 'stance['+sender+']['+recipient +']=' +str(stance) + message
                        msg = Message(sender=recipient,
                                    recipient=sender,
                                    message=message,
                                    phase=dip_game.game.get_current_phase())
                        dip_game.new_message(msg)

                        if answer =='YES':
                            orders[recipient]= dip_game.proposal_received[recipient][sender] + orders[recipient]
                            proposal_stat.append([dip_step, dip_player.bot_type[sender], sender, recipient, 1, centers[sender], centers[recipient]])
                        else:
                            proposal_stat.append([dip_step, dip_player.bot_type[sender], sender, recipient, 0, centers[sender], centers[recipient]])
        
        
        for power_name, power_orders in orders.items():
            dip_game.game.set_orders(power_name, power_orders)

        for power in dip_game.powers:
            dip_player.update_stance(dip_game.game, power)

        dip_game.game_process()
        if dip_game.game.phase_type != 'A' and dip_game.game.phase_type != 'R':
            for i in range (last_ep_index, len(env.ep_states)):
                state, sender, recipient, one_hot_order = env.ep_info[i]
                #reward = self + ally supply center
                #find all allies 
                sender_reward = len(dip_game.game.get_centers(sender)) - centers[sender]
                sender_stance =  dip_player.stance[sender]
                ally_reward = 0

                for power in sender_stance:
                    if sender_stance[power] > 1 and power!=sender:
                        ally_reward += (len(dip_game.game.get_centers(power)) - centers[power]) 
                        # if order_info[0]=='self' and order_info[1]=='support' and order_info[2]=='ally':
                        #     sender_reward += len(dip_game.game.get_centers(power)) * DISCOUNT_ORDER_REWARD
                    # if sender_stance[power] < -1 and power!=sender and one_hot_order:
                    #     order_info = env.index_order(one_hot_order, 'str') # if power is the enemy, penalty if send self attack order to enemy [0,3,3]
                    #     # print(order_info)
                    #     if order_info[0]=='self' and order_info[1]=='attack' and order_info[2]=='enemy':
                    #         sender_reward -= len(dip_game.game.get_centers(power))* DISCOUNT_ORDER_REWARD

                sender_reward += ally_reward * DISCOUNT_ALLY_REWARD
                if state=='no_more_order':
                    env.ep_states[i][env.power_mapping[sender]][0] = dip_player.stance[sender][recipient]
                if state=='censoring': #update stance of next states of states = share order/do not share order
                    # env.ep_n_states[i][env.power_mapping[sender]][0] = dip_player.stance[sender][recipient]
                    if env.ep_actions[i][env.power_mapping[sender]]==1:# set reward for sharing order 
                        env.ep_rewards.append({id: sender_reward*1. if id ==env.power_mapping[sender] else 0. for id in env.agent_id})   
                    else:
                        env.ep_rewards.append({id: 0. for id in env.agent_id})   
                    
            last_ep_index = len(env.ep_states)
        dip_step +=1
    if dip_game.game.is_game_done:
    
        centers = {power: len(dip_game.game.get_centers(power)) for power in dip_game.powers}
        env.ep_rewards.append({id: centers[power] for power,id in env.power_mapping.items()}) 
        maa2c.n_episodes += 1
        maa2c.episode_done = True

    rewards = arr_dict_to_arr(env.ep_rewards, N_AGENTS)
    rewards = np.array(rewards)
    
    # if AGENT_VERSION == 'v2':
    #     save_to_json(hist_name, dip_game, dip_player.bot_type, None)
    # else:
    save_to_json(hist_name, dip_game, dip_player.bot_type, proposal_stat, stance_rewards)
    EVAL_REWARDS = rewards
    stop_io_loop()


@gen.coroutine  
def orders_of_generated_game(current_game, player, power):
    has_shared_orders = False
    generated_game = current_game.game.__deepcopy__(None) 

    centers = {power: len(generated_game.get_centers(power)) for power in generated_game.powers}    # rank powers by current supply center
    centers[power] = -1 # set itself to has least supply centers 
    sorted_powers = [power for power,n in sorted(centers.items(), key=lambda item: item[1], reverse=True)]
    
    sorted_powers.pop() # remove last index or itself from a sorted list
    # print('we are: ', power)
    
    for other_power in sorted_powers:
        other_power_orders = current_game.received[power][other_power]
        # print('considering shared orders: ', other_power)
        # print(other_power_orders)
        if other_power_orders:
            generated_game.set_orders(other_power, other_power_orders)
            has_shared_orders = True
    curr_phase = current_game.game.get_current_phase()
    if has_shared_orders:
        generated_game.process()
        generated_game.set_current_phase(curr_phase)
        
    orders = yield player.get_orders(generated_game, power)
    return orders

def save_to_json(name, game, bot_type, proposal_stat, stance_rewards):
    game_history_name = name + '_with_baseline_bots_1RLvs6Transparent_proposal_{}'.format(EPISODE+1) 
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
            # if AGENT_VERSION == 'v2':
            #     phase['current_state_order'] = order_game_memo
            # header = phase.keys()
            csv_writer.writerow(header)
            count += 1

        # Writing data of CSV file
        csv_writer.writerow(phase.values())

    data_file.close()

    # dict_stat_header = ['sender', 'recipient','share','power1', 'order_type','power2','stance of recipient']
    # data_file = open(exp + '_stat.csv', 'w')
    # csv_writer = csv.writer(data_file)
    # csv_writer.writerow(dict_stat_header)
    # csv_writer.writerows(dict_stat)
    # data_file.close()
    

    # data_file = open(exp + '_reward.csv', 'w')
    # csv_writer = csv.writer(data_file)
    # csv_writer.writerows(rewards)
    # data_file.close()

    # [dip_step, dip_player.bot_type[sender], sender, recipient, 0, centers[sender], centers[recipient]]
    proposal_stat_header = ['step', 'bot type', 'sender', 'recipient', 'accept?', 'number of sender sc', 'number of recipient sc']
    data_file = open(exp + '_proposal_stat.csv', 'w')
    csv_writer = csv.writer(data_file)
    csv_writer.writerow(proposal_stat_header)
    csv_writer.writerows(proposal_stat)
    data_file.close()


    # [dip_step, dip_player.bot_type[power], power, centers[power], ally_power, ally, enemy_power, enemy]
    stance_rewards_header = ['step', 'bot type','power','number of sc', 'ally power','number of ally sc','enemy power','number of enemy sc']
    data_file = open(exp + '_phase_rewards.csv', 'w')
    csv_writer = csv.writer(data_file)
    csv_writer.writerow(stance_rewards_header)
    csv_writer.writerows(stance_rewards)
    data_file.close()

# @gen.coroutine
def main():    
    global EPISODE
    for i in range(TEST_EPISODES):
        start_io_loop(test)
        EPISODE +=1
  
        
if __name__ == '__main__':
    main()
