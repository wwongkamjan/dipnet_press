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

MAX_EPISODES = 2
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 1
EVAL_INTERVAL = 1
DISCOUNT_ALLY_REWARD = 0.7
DISCOUNT_ORDER_REWARD = 0.3
LOAD_MODEL = True

# roll out n steps
ROLL_OUT_N_STEPS = 40
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
# v1=train comm agents but no press tactics for the orders 
# v2=train comm agents with press tactics for the orders 

RANDOM_SEED = 2017
N_AGENTS = 7
K_ORDERS = 5
AGENT = None
EVAL_REWARDS = None

@gen.coroutine
def interact():

    env = DiplomacyEnv()
    global AGENT

    state_dim = env.observation_space.shape[0]
    if len(env.action_space.shape) > 1:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    if AGENT==None:
        maa2c = MAA2C(env=env, n_agents=N_AGENTS, 
                state_dim=state_dim, action_dim=action_dim, memory_capacity=MEMORY_CAPACITY,
                batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                done_penalty=DONE_PENALTY, roll_out_n_steps=ROLL_OUT_N_STEPS,
                reward_gamma=REWARD_DISCOUNTED_GAMMA,
                epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
                episodes_before_train=EPISODES_BEFORE_TRAIN, training_strategy="centralized",
                critic_loss=CRITIC_LOSS, actor_parameter_sharing=True, critic_parameter_sharing=True) 
        if LOAD_MODEL: 
            maa2c.load_model('models/a2c_actor_diplomacy_{}'.format(AGENT_VERSION), 'models/a2c_critic_diplomacy_{}'.format(AGENT_VERSION))
    else:
        maa2c = AGENT   
        maa2c.env_state = dict_to_arr(env.reset(), N_AGENTS)
    dip_step = 0

    # if (maa2c.max_steps is not None) and (maa2c.n_steps >= maa2c.max_steps):
    #     maa2c.env_state = dict_to_arr(env.reset(), N_AGENTS)
    dip_game = env.dip_game
    dip_player = env.dip_player
    last_ep_index = 0
    while not dip_game.game.is_game_done and dip_step < ROLL_OUT_N_STEPS:
        if dip_game.game.phase_type != 'A' and dip_game.game.phase_type != 'R':
            centers = {power: len(dip_game.game.get_centers(power)) for power in dip_game.powers}
            for sender in dip_game.powers:
                for recipient in dip_game.powers:
                    share_order_list = []
                    if sender != recipient and not dip_game.powers[sender].is_eliminated() and not dip_game.powers[recipient].is_eliminated():
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

        orders = yield {power_name: dip_player.get_orders(dip_game.game, power_name) for power_name in dip_game.powers}
        #generating an imagined world from received messages
        if AGENT_VERSION == 'v2':
            new_orders = yield {power_name: orders_of_generated_game(dip_game, dip_player, power_name) for power_name in dip_game.powers}
        
            # print('new_orders: ', new_orders)
            # print('orders: ', orders)
            orders =new_orders

        for power_name, power_orders in orders.items():
            dip_game.game.set_orders(power_name, power_orders)

        for power in dip_game.powers:
            dip_player.update_stance(dip_game.game, power)

        dip_game.game_process()
        # print('game process')
        # update next state list and get reward from result of the phase 


        if dip_game.game.phase_type != 'A' and dip_game.game.phase_type != 'R':
            for i in range (last_ep_index, len(env.ep_states)):
                state, sender, recipient, one_hot_order = env.ep_info[i]
                #reward = self + ally supply center
                #find all allies 
                sender_reward = len(dip_game.game.get_centers(sender)) - centers[sender]
                sender_stance =  dip_player.stance[sender]
                for power in sender_stance:
                    if sender_stance[power] > 1 and power!=sender:
                        sender_reward += (len(dip_game.game.get_centers(power)) - centers[power]) * DISCOUNT_ALLY_REWARD
                        if order_info[0]=='self' and order_info[1]=='support' and order_info[2]=='ally':
                            sender_reward += len(dip_game.game.get_centers(power)) * DISCOUNT_ORDER_REWARD
                    if sender_stance[power] < -1 and power!=sender and one_hot_order:
                        order_info = env.index_order(one_hot_order, 'str') # if power is the enemy, penalty if send self attack order to enemy [0,3,3]
                        # print(order_info)
                        if order_info[0]=='self' and order_info[1]=='attack' and order_info[2]=='enemy':
                            sender_reward -= len(dip_game.game.get_centers(power))* DISCOUNT_ORDER_REWARD
 
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
        
    if dip_game.game.is_game_done or dip_step >= ROLL_OUT_N_STEPS:
        
        centers = {power: len(dip_game.game.get_centers(power)) for power in dip_game.powers}
        final_r = [0.0] * maa2c.n_agents
        for power in dip_game.powers:
            final_r[env.power_mapping[power]] = centers[power]
        maa2c.n_episodes += 1
        maa2c.episode_done = True

    # print('check dict of states: ', env.ep_states[-10:])    
    #tranform s,a,r from dict to arr
    #wrong -> env.* = array of dict
    # next_states = arr_dict_to_arr(env.ep_n_states, N_AGENTS)
    rewards = arr_dict_to_arr(env.ep_rewards, N_AGENTS)
    # dones = arr_dict_to_arr(env.ep_dones, N_AGENTS)
    actions = arr_dict_to_arr(env.ep_actions, N_AGENTS)
    states = arr_dict_to_arr(env.ep_states, N_AGENTS)


    # print('check states: ', states[-10:])
    
    for i in range (len(actions)):
        new_arr = []
        new_arr = [index_to_one_hot(a, action_dim) for a in actions[i]]
        actions[i] = new_arr
        
    # print('check actions: ', actions[-10:])
    rewards = np.array(rewards)
    print(rewards[:,0])
    for agent_id in range(maa2c.n_agents):
        rewards[:,agent_id] = maa2c._discount_reward(rewards[:,agent_id], final_r[agent_id])
    rewards = rewards.tolist()
    # print('check rewards: ', rewards[-10:])
    maa2c.n_steps += 1
    
    maa2c.memory.push(states, actions, rewards)

    maa2c.env_state = dict_to_arr(env.reset(), N_AGENTS)

    AGENT = maa2c
    print('Done collecting experience')
    stop_io_loop()


@gen.coroutine   
def evaluation():
    global EVAL_REWARDS
    hist_name = 'comm_agent_{}'.format(AGENT_VERSION)
    env = DiplomacyEnv()
    rewards = []
    maa2c = AGENT
    for i in range(EVAL_EPISODES):
        maa2c.env_state = dict_to_arr(env.reset(), N_AGENTS)
        dip_game = env.dip_game
        dip_player = env.dip_player
        last_ep_index = 0
        order_game_memo= {}
        while not dip_game.game.is_game_done:
            if dip_game.game.phase_type != 'A' and dip_game.game.phase_type != 'R':
                centers = {power: len(dip_game.game.get_centers(power)) for power in dip_game.powers}
                for sender in dip_game.powers:
                    for recipient in dip_game.powers:
                        share_order_list=[]
                        if sender != recipient and not dip_game.powers[sender].is_eliminated() and not dip_game.powers[recipient].is_eliminated():
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
                            message = 'stance[{}][{}]= {} '.format(sender, recipient, str(stance)) + message
                            msg = Message(sender=sender,
                                        recipient=recipient,
                                        message=message,
                                        phase=dip_game.game.get_current_phase())
                            dip_game.new_message(msg)

              
            orders = yield {power_name: dip_player.get_orders(dip_game.game, power_name) for power_name in dip_game.powers}
            if AGENT_VERSION == 'v2':
                new_orders = yield {power_name: orders_of_generated_game(dip_game, dip_player, power_name) for power_name in dip_game.powers}
            
                # print('new_orders: ', new_orders)
                # print('orders: ', orders)
                order_game_memo[dip_game.game._phase_wrapper_type(dip_game.game.current_short_phase)] = orders
                orders =new_orders
            for power_name, power_orders in orders.items():
                dip_game.game.set_orders(power_name, power_orders)

            for power in dip_game.powers:
                dip_player.update_stance(dip_game.game, power)
            dip_game.game_process()
            # print('game process')
            # update next state list and get reward from result of the phase 


            if dip_game.game.phase_type != 'A' and dip_game.game.phase_type != 'R':
                for j in range (last_ep_index, len(env.ep_states)):
                    state, sender, recipient, one_hot_order = env.ep_info[i]
                    #reward = self + ally supply center
                    #find all allies 
                    sender_reward = 0
                    sender_stance =  dip_player.stance[sender]
                    for power in sender_stance:
                        if sender_stance[power] > 1 or power==sender:
                            sender_reward += len(dip_game.game.get_centers(power)) - centers[power]
                    if state=='no_more_order':
                        env.ep_states[j][env.power_mapping[sender]][0] = dip_player.stance[sender][recipient]
                    if state=='censoring': #update stance of next states of states = share order/do not share order
                       
                        if env.ep_actions[j][env.power_mapping[sender]]==1:# set reward for sharing order 
                            env.ep_rewards.append({id: sender_reward*1. if id ==env.power_mapping[sender] else 0. for id in env.agent_id})   
                        else:
                            env.ep_rewards.append({id: 0. for id in env.agent_id})   
                        
                last_ep_index = len(env.ep_states)
            
        if dip_game.game.is_game_done:
            
            centers_id = {id: len(dip_game.game.get_centers(power)) for power, id in env.power_mapping.items()}
            env.ep_rewards.append({id: centers_id[id]*1. for id in env.agent_id})   

        rewards.append(arr_dict_to_arr(env.ep_rewards, N_AGENTS))
        print('evaluation result: ' )
        for power,id in env.power_mapping.items():
            print('%s: %d centers' %(power, centers_id[id]))
        
        
        
        if AGENT_VERSION == 'v1':
            save_to_json(hist_name, maa2c.n_episodes, i, dip_game, None)
        else:
            save_to_json(hist_name, maa2c.n_episodes, i, dip_game, order_game_memo)
    # maa2c.save_model(actor_path='models/a2c_actor_diplomacy_{}'.format(AGENT_VERSION), critic_path = 'models/a2c_critic_diplomacy_{}'.format(AGENT_VERSION))
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
    # print('considering shared orders: ', sorted_powers)
    for other_power in sorted_powers:
        other_power_orders = current_game.received[power][other_power]
        if other_power_orders:
            generated_game.set_orders(other_power, other_power_orders)
            has_shared_orders = True
    curr_phase = current_game.game.get_current_phase()
    if has_shared_orders:
        generated_game.process()
        generated_game.set_current_phase(curr_phase)
        
    orders = yield player.get_orders(generated_game, power)
    return orders


def save_to_json(name, ep, eval_i, game, order_game_memo):
    game_history_name = name + '_eval_episode_' +str(ep)+ '_'+str(eval_i)
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
    count = 0
    for phase in game_data:
        if count == 0:
            # Writing headers of CSV file
            if AGENT_VERSION == 'v2':
                phase['current_state_order'] = order_game_memo
            header = phase.keys()

            csv_writer.writerow(header)
            count += 1

        # Writing data of CSV file
        csv_writer.writerow(phase.values())

    data_file.close()
        
# @gen.coroutine
def main():    
    episodes =[]
    eval_rewards =[]
    while AGENT==None or AGENT.n_episodes < MAX_EPISODES:
        print('interact')
        start_io_loop(interact)
        if AGENT.n_episodes >= EPISODES_BEFORE_TRAIN:
            print('train')
            AGENT.train()
        if AGENT.episode_done and ((AGENT.n_episodes+1)%EVAL_INTERVAL == 0):
            print('evaluate')
            start_io_loop(evaluation)
            rewards = EVAL_REWARDS
            # print(rewards)
            rewards_mu, rewards_std = ma_agg_double_list(rewards)
            
            for agent_id in range (N_AGENTS):
                print("Episode %d, Agent %d, Average Reward %.2f" % (AGENT.n_episodes+1, agent_id, rewards_mu[agent_id]))
            episodes.append(AGENT.n_episodes+1)
            eval_rewards.append(rewards_mu)
  
        
if __name__ == '__main__':
    main()
