from pytorch_DRL.MAA2C import MAA2C
from pytorch_DRL.common.utils import ma_agg_double_list, dict_to_arr, arr_dict_to_arr, index_to_one_hot
from tornado import gen
import ujson as json
import csv
import random

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("../..")
sys.path.append("../../baseline_bots")
import numpy as np
import matplotlib.pyplot as plt
import random

# MAA2C: https://github.com/ChenglongChen/pytorch-DRL/
from DiplomacyEnv import DiplomacyEnv
from diplomacy.engine.message import Message
from diplomacy.utils.export import to_saved_game_format
from diplomacy_research.utils.cluster import start_io_loop, stop_io_loop
from DAIDE import ORR, XDO
from baseline_bots.bots.pushover_bot import PushoverBot
from baseline_bots.bots.random_no_press import RandomNoPressBot
from baseline_bots.bots.dipnet.no_press_bot import NoPressDipBot
from baseline_bots.bots.dipnet.RealPolitik import RealPolitik


MAX_EPISODES = 100
EPISODES_BEFORE_TRAIN = 2
TEST_EPISODES = 100
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
# AGENT_VERSION = "v2" 

RANDOM_SEED = 1000
BOTS = ['RL', 'dipnet', 'dipnet','dipnet', 'dipnet', 'dipnet', 'pushover']
N_AGENTS = 7
K_ORDERS = 5
AGENT = None
EVAL_REWARDS = None

@gen.coroutine
def test():
    global EVAL_REWARDS
    global AGENT
    global EPISODE
    hist_name = 'proposer_agent'
    env = DiplomacyEnv()
    env.n_agents = N_AGENTS
    env.agent_id = [id for id in range(env.n_agents)]
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

    maa2c.load_model('models/a2c_actor_diplomacy_proposer', 'models/a2c_critic_diplomacy_proposer')

    dip_step = 0

    dip_game = env.dip_game
    dip_player = env.dip_player
    dip_player.bot_type = {power: b for b,power in zip(BOTS, dip_game.powers)}
    bot_instance = {power: None for power in dip_game.powers}
    agent_id = 0
    for power,bot in dip_player.bot_type.items():
        if bot != 'RL':
            if  bot== 'pushover':
                bot_instance[power] = PushoverBot(power,dip_game.game)
            elif  bot == 'dipnet':
                bot_instance[power] = NoPressDipBot(power,dip_game.game)
            elif  bot == 'random':
                bot_instance[power] = RandomNoPressBot(power,dip_game.game)
            elif  bot == 'rplt':
                bot_instance[power] = RealPolitik(power,dip_game.game)
        else:
            env.power_mapping[power] = agent_id
        agent_id += 1
    RL_agent_id = env.power_mapping.values()
    last_ep_index = 0
    
    propose_data = False

    while not dip_game.game.is_game_done:
        orders = yield {power_name: dip_player.get_orders(dip_game.game, power_name) for power_name in dip_game.powers}
        if dip_game.game.phase_type =='M':
            centers = {power: len(dip_game.game.get_centers(power)) for power in dip_game.powers}
            propose_data = True
            for sender in dip_game.powers:

                if  dip_player.bot_type[sender] == 'pushover':

                    # Retrieve messages
                    rcvd_messages = dip_game.game.filter_messages(messages=dip_game.game.messages, game_role=sender)
                    rcvd_messages = list(rcvd_messages.items())
                    rcvd_messages.sort()
                    rcvd_messages = [msg for _,msg in rcvd_messages]

                    p_bot = bot_instance[sender]
                    return_obj = p_bot(rcvd_messages)
                    for msg in return_obj['messages']:
                            msg_obj = Message(
                                sender=sender,
                                recipient=msg['recipient'],
                                message=msg['message'],
                                phase=dip_game.game.get_current_phase(),
                            )
                            dip_game.game.add_message(message=msg_obj)
                elif dip_player.bot_type[sender] == 'rplt':
                    # Retrieve messages
                    rcvd_messages = dip_game.game.filter_messages(messages=dip_game.game.messages, game_role=sender)
                    rcvd_messages = list(rcvd_messages.items())
                    rcvd_messages.sort()
                    return_obj = yield bot_instance[sender].gen_messages(rcvd_messages)
                    for msg in return_obj:
                        msg_obj = Message(
                            sender=sender,
                            recipient=msg['recipient'],
                            message=msg['message'],
                            phase=dip_game.game.get_current_phase(),
                        )
                        dip_game.game.add_message(message=msg_obj)
                elif dip_player.bot_type[sender] == 'RL':
                    propose_data = True
                    for recipient in dip_game.powers:
                        proposal_list = []
                        if sender != recipient and not dip_game.powers[sender].is_eliminated() and not dip_game.powers[recipient].is_eliminated():
                            orders = yield dip_player.get_orders(dip_game.game, recipient)
                            stance = dip_player.stance[sender][recipient] 
                            env.set_power_state(sender, stance)
                            n = len(orders)
                            for order in orders[:min(K_ORDERS,n)]:
                                #state = no order in consideation
                                maa2c.env_state = dict_to_arr(env.cur_obs, N_AGENTS)
                                action = maa2c.exploration_action(maa2c.env_state)
                                action_dict = {agent_id: action[agent_id] if agent_id == env.power_mapping[sender] else 0 for agent_id in env.agent_id}
                                env.step(action_dict, sender, recipient, order)

                                #state = considering order
                                maa2c.env_state = dict_to_arr(env.cur_obs, N_AGENTS)
                                action = maa2c.exploration_action(maa2c.env_state)
                                action_dict = {agent_id: action[agent_id] if agent_id == env.power_mapping[sender] else 0 for agent_id in env.agent_id}
                                env.step(action_dict, sender, recipient, order)
                                # if action=propose, we add it to the list
                                if action_dict[env.power_mapping[sender]]==1:
                                    env.step(action_dict, sender, recipient, order)
                                    proposal_list.append(order)
                                        
                            if len(proposal_list)>0:
                                suggested_orders = ORR([XDO(order) for order in proposal_list])
                                msg_obj = Message(
                                    sender=sender,
                                    recipient=recipient,
                                    message=str(suggested_orders),
                                    phase=dip_game.game.get_current_phase(),
                                )
                                dip_game.game.add_message(message=msg_obj)
                            env.reset_power_state(sender, recipient)

        orders = {}
        for power,bot in dip_player.bot_type.items():
            if bot == 'RL':
                orders[power] = yield dip_player.get_orders(dip_game.game, power)
            elif bot == 'dipnet' or bot == 'rplt':
                orders[power] = yield bot_instance[power].gen_orders()
            else:
                orders[power] = bot_instance[power].gen_orders()
                        
        for power_name, power_orders in orders.items():
            # print(power_name)
            # print(power_orders)
            dip_game.game.set_orders(power_name, power_orders)

        for power in dip_game.powers:
            dip_player.update_stance(dip_game.game, power)

        dip_game.game_process()


        if propose_data:
            for i in range (last_ep_index, len(env.ep_states)):
                state, sender, recipient, one_hot_order = env.ep_info[i]
                #reward = self + ally supply center
                #find all allies 
                sender_reward = len(dip_game.game.get_centers(sender)) - centers[sender]
                sender_stance =  dip_player.stance[sender]
                for power in sender_stance:
                    if sender_stance[power] > 1 and power!=sender:
                        sender_reward += (len(dip_game.game.get_centers(power)) - centers[power]) * DISCOUNT_ALLY_REWARD
 
                if state=='no_more_order':
                    env.ep_states[i][env.power_mapping[sender]][0] = dip_player.stance[sender][recipient]
                if state=='censoring': #update stance of next states of states = share order/do not share order
                    # env.ep_n_states[i][env.power_mapping[sender]][0] = dip_player.stance[sender][recipient]
                    if env.ep_actions[i][env.power_mapping[sender]]==1:# set reward for sharing order 
                        env.ep_rewards.append({id: sender_reward*1. if id ==env.power_mapping[sender] else 0. for id in env.agent_id})   
                    else:
                        env.ep_rewards.append({id: sender_reward*-1. if id ==env.power_mapping[sender] else 0. for id in env.agent_id})   
                    
            last_ep_index = len(env.ep_states)
            propose_data = False
        dip_step +=1

    if dip_game.game.is_game_done:
    
        centers_id = {id: len(dip_game.game.get_centers(power)) for power, id in env.power_mapping.items()}
        env.ep_rewards.append({id: centers_id[id]*1. if id in RL_agent_id else 0. for id in env.agent_id}) 
        maa2c.n_episodes += 1
        maa2c.episode_done = True

    rewards = arr_dict_to_arr(env.ep_rewards, N_AGENTS)
    rewards = np.array(rewards)
    
    save_to_json(hist_name, dip_game, dip_player.bot_type)
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

def save_to_json(name, game, bot_type):
    game_history_name = name + '_with_baseline_bots_{}'.format(EPISODE+1) 
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


# @gen.coroutine
def main():    
    global EPISODE
    for i in range(TEST_EPISODES):
        start_io_loop(test)
        EPISODE +=1
  
        
if __name__ == '__main__':
    main()
