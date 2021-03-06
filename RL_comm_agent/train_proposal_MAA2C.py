from pytorch_DRL.MAA2C import MAA2C
from pytorch_DRL.common.utils import ma_agg_double_list, dict_to_arr, arr_dict_to_arr, index_to_one_hot
from tornado import gen
import ujson as json
import csv
from test_proposal_MAA2C import test

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

MAX_EPISODES = 50
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 1
EVAL_INTERVAL = 5
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
# AGENT_VERSION = "v2" 
# v1=train comm agents but no press tactics for the orders 
# v2=train comm agents with press tactics for the orders 

RANDOM_SEED = 2020
BOTS = ['pushover', 'RL', 'pushover', 'random', 'dipnet', 'pushover', 'pushover']
N_AGENTS = 7 # relative to number of 'RL' bots in BOTS
K_ORDERS = 5
AGENT = None
EVAL_REWARDS = None


@gen.coroutine
def interact():

    env = DiplomacyEnv()
    global AGENT
    env.n_agents = N_AGENTS
    env.agent_id = [id for id in range(env.n_agents)]
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
            maa2c.load_model('models/a2c_actor_diplomacy_proposer', 'models/a2c_critic_diplomacy_proposer')
    else:
        maa2c = AGENT   
        maa2c.env_state = dict_to_arr(env.reset(), N_AGENTS)
    dip_step = 0

    dip_game = env.dip_game
    dip_player = env.dip_player
    dip_player.bot_type = {power: b for b,power in zip(BOTS, dip_game.powers)}
    print(dip_player.bot_type)
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
        else:
            #get agent_id of RL agent
            env.power_mapping[power] = agent_id
        agent_id += 1
    RL_agent_id = env.power_mapping.values()
    print(env.power_mapping)
    last_ep_index = 0
    propose_data = False
    while not dip_game.game.is_game_done and dip_step < ROLL_OUT_N_STEPS:
        if dip_game.game.phase_type =='M':
            centers = {power: len(dip_game.game.get_centers(power)) for power in dip_game.powers}
            for sender in dip_game.powers:
                if  dip_player.bot_type[sender] == 'pushover':
                    # Retrieve messages
                    rcvd_messages = dip_game.game.filter_messages(messages=dip_game.game.messages, game_role=sender)
                    rcvd_messages = list(rcvd_messages.items())
                    rcvd_messages.sort()
                    # print(rcvd_messages[0])
                    rcvd_messages = [msg for _,msg in rcvd_messages]

                    p_bot = bot_instance[sender]
                    return_obj = p_bot(rcvd_messages)
                    # print(return_obj)
                    for msg in return_obj['messages']:
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
                                # print('new order', env.cur_obs)
                                maa2c.env_state = dict_to_arr(env.cur_obs, N_AGENTS)
                                action = maa2c.exploration_action(maa2c.env_state)
                                action_dict = {agent_id: action[agent_id] if agent_id == env.power_mapping[sender] else 0 for agent_id in env.agent_id}
                                env.step(action_dict, sender, recipient, order)

                                #state = considering order
                                maa2c.env_state = dict_to_arr(env.cur_obs, N_AGENTS)
                                action = maa2c.exploration_action(maa2c.env_state)
                                action_dict = {agent_id: action[agent_id] if agent_id == env.power_mapping[sender] else 0 for agent_id in env.agent_id}
                                # print(action_dict)
                                env.step(action_dict, sender, recipient, order)
                                # if action=propose, we add it to the list
                                # print('internal step',env.cur_obs)
                                if action_dict[env.power_mapping[sender]]==1:
                                    env.step(action_dict, sender, recipient, order)
                                    # print('to propose', env.cur_obs)
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
                            # print('no more order for this recipient',env.cur_obs)

        orders = {}
        for power,bot in dip_player.bot_type.items():
            if bot == 'RL':
                orders[power] = yield dip_player.get_orders(dip_game.game, power)
            elif bot == 'dipnet':
                orders[power] = yield bot_instance[power].gen_orders()
            else:
                orders[power] = bot_instance[power].gen_orders()

        for power_name, power_orders in orders.items():
            dip_game.game.set_orders(power_name, power_orders)

        for power in dip_game.powers:
            dip_player.update_stance(dip_game.game, power)

        dip_game.game_process()
        # print('game process')
        # update next state list and get reward from result of the phase 

        if propose_data:
            for i in range (last_ep_index, len(env.ep_states)):
                state, sender, recipient, one_hot_order = env.ep_info[i]
                #reward = self + ally supply center
                #find all allies 
                sender_reward = len(dip_game.game.get_centers(sender)) - centers[sender] + len(dip_game.game.powers[sender].units)
                sender_stance =  dip_player.stance[sender]
                for power in sender_stance:
                    if sender_stance[power] > 1 and power!=sender:
                        sender_reward += (len(dip_game.game.get_centers(power)) - centers[power] + len(dip_game.game.powers[sender].units)) * DISCOUNT_ALLY_REWARD
 
                if state=='no_more_order':
                    env.ep_states[i][env.power_mapping[sender]][0] = dip_player.stance[sender][recipient]
                    env.ep_rewards.append({id: 0. for id in env.agent_id})
                elif state=='censoring': #update stance of next states of states = share order/do not share order
                    # env.ep_n_states[i][env.power_mapping[sender]][0] = dip_player.stance[sender][recipient]
                    if env.ep_actions[i][env.power_mapping[sender]]==1:# set reward for sharing order 
                        env.ep_rewards.append({id: sender_reward*1. if id ==env.power_mapping[sender] else 0. for id in env.agent_id})   
                    else:
                        env.ep_rewards.append({id: sender_reward*-1. if id ==env.power_mapping[sender] else 0. for id in env.agent_id})   
                else:
                    env.ep_rewards.append({id: 0. for id in env.agent_id})
            last_ep_index = len(env.ep_states)
            propose_data = False
        dip_step +=1
        
    if dip_game.game.is_game_done or dip_step >= ROLL_OUT_N_STEPS:
        
        centers = {power: len(dip_game.game.get_centers(power)) for power in dip_game.powers}
        final_r = [0.0] * env.n_agents
        for power in dip_game.powers:
            if dip_player.bot_type[power] == 'RL':
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
    print(len(rewards))
    print(len(actions))
    print(len(states))
    # print('check states: ', states[-10:])
    
    for i in range (len(actions)):
        new_arr = []
        new_arr = [index_to_one_hot(a, action_dim) for a in actions[i]]
        actions[i] = new_arr
        
    # print('check actions: ', actions[-10:])
    rewards = np.array(rewards)

    for agent_id in RL_agent_id:
        # print('{} reward: {}'.format(agent_id, np.sum(rewards[:,agent_id])))
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
        
# @gen.coroutine
def main():    

    while AGENT==None or AGENT.n_episodes < MAX_EPISODES:
        print('interact')
        start_io_loop(interact)
        if AGENT.n_episodes >= EPISODES_BEFORE_TRAIN:
            print('train')
            AGENT.train()
            AGENT.save_model(actor_path='models/a2c_actor_diplomacy_proposer', critic_path = 'models/a2c_critic_diplomacy_proposer')
        if AGENT.episode_done and ((AGENT.n_episodes+1)%EVAL_INTERVAL == 0):
            print('evaluate')
            # start_io_loop(evaluation)
            # rewards = EVAL_REWARDS
            # print(rewards)
            # rewards_mu, rewards_std = ma_agg_double_list(rewards)
            
            # for agent_id in range (N_AGENTS):
            #     print("Episode %d, Agent %d, Average Reward %.2f" % (AGENT.n_episodes+1, agent_id, rewards_mu[agent_id]))
            # episodes.append(AGENT.n_episodes+1)
            # eval_rewards.append(rewards_mu)
            start_io_loop(test)
  
        
if __name__ == '__main__':
    main()
