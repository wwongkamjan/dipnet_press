from pytorch_DRL.MAA2C import MAA2C
from pytorch_DRL.common.utils import ma_agg_double_list
from tornado import gen


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import random

# MAA2C: https://github.com/ChenglongChen/pytorch-DRL/
from DiplomacyEnv import DiplomacyEnv
from diplomacy_research.utils.cluster import start_io_loop, stop_io_loop

MAX_EPISODES = 10
EPISODES_BEFORE_TRAIN = 5
EVAL_EPISODES = 10
EVAL_INTERVAL = 2

# roll out n steps
ROLL_OUT_N_STEPS = 100
# only remember the latest 2 ROLL_OUT_N_STEPS
MEMORY_CAPACITY = 2*ROLL_OUT_N_STEPS
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
K_ORDERS = 10

@gen.coroutine
def interact(env, maa2c):
    dip_step = 0
    if (maa2c.max_steps is not None) and (maa2c.n_steps >= maa2c.max_steps):
        # env_state is dictionary
        maa2c.env_state = self.env.reset()
        # tranfrom from dict to arr
        maa2c.env_state = maa2c.agentdict_to_arr(maa2c.env_state)
    dip_game = env.dip_game
    dip_player = env.dip_player
    last_ep_index = 0
    while not dip_game.game.is_game_done:
        centers = {power: len(dip_game.game.get_centers(power)) for power in dip_game.powers}
        for sender in dip_game.powers:
            for recipient in dip_game.powers:
                if sender != recipient and not dip_game.powers[sender].is_eliminated() and not dip_game.powers[recipient].is_eliminated():
                    orders = yield dip_player.get_orders(dip_game.game, sender)
                    stance = dip_player.stance[sender][recipient] 
                    n = len(orders)
                    for order in orders[:min(K_ORDERS,n)]:
                        print(order)
                        env.set_power_state(sender, stance)
                        maa2c.env_state = maa2c.agentdict_to_arr(env.cur_obs)
                        action = maa2c.exploration_action(maa2c.env_state)
                        action_dict = {agent_id: action[agent_id] for agent_id in range(maa2c.n_agents)}
                        env.step(action_dict, sender, recipient, order)
                        
                env.reset_power_state(sender, recipient)

        orders = yield {power_name: dip_player.get_orders(dip_game.game, power_name) for power_name in dip_game.powers}
        for power_name, power_orders in orders.items():
           dip_game.game.set_orders(power_name, power_orders)

        dip_game.game_process()
        
        # update next state list and get reward from result of the phase 
        for power in dip_game.powers:
            dip_player.update_stance(dip_game, power)
        
        for i in range (last_ep_index, len(env.ep_n_states)):
            state, sender, recipient, one_hot_order = env.ep_info[i]
            #reward = self + ally supply center
            #find all allies 
            sender_reward = 0
            sender_stance =  dip_player.stance[sender]
            for power in sender_stance:
                if sender_stance[power] > 1 or power==sender:
                    sender_reward += len(dip_game.game.get_centers(power)) - centers[power]
            if state=='no_more_order':
                env.ep_states[i][sender][0] = dip_player.stance[sender][recipient]
            if state=='censoring': #update stance of next states of states = share order/do not share order
                env.ep_n_states[i][sender][0] = dip_player.stance[sender][recipient]
                if env.ep_actions[i][env.power_mapping[sender]]==1:# set reward for sharing order 
                    env.ep_rewards.append({id: sender_reward if id ==env.power_mapping[sender] else 0 for id in env.agent_id})   
                else:
                    env.ep_rewards.append({id: 0 for id in env.agent_id})   
                    
        last_ep_index = len(env.ep_n_states) -1
        dip_step +=1
        
    if dip_game.game.is_game_done:
        maa2c.env_state = env.reset()
        # tranfrom from dict to arr
        maa2c.env_state = maa2c.agentdict_to_arr(maa2c.env_state)
        centers = {power: len(dip_game.get_centers[power]) for power in dip_game.powers}
        final_r = [0.0] * maa2c.n_agents
        for power in dip_game.powers():
            final_r[env.power_mapping[power]] = len(centers[power])
        maa2c.n_episodes += 1
        maa2c.episode_done = True
        
    #tranform s,a,r from dict to arr
    next_states = maa2c.agentdict_to_arr(ep_n_states)
    rewards = maa2c.agentdict_to_arr(ep_rewards)
    dones = maa2c.agentdict_to_arr(ep_dones)
    actions = maa2c.agentdict_to_arr(ep_actions)
    states = maa2c.agentdict_to_arr(ep_states)
    
    rewards = np.array(rewards)
    for agent_id in range(maa2c.n_agents):
        rewards[:,agent_id] = maa2c._discount_reward(rewards[:,agent_id], final_r[agent_id])
    rewards = rewards.tolist()
    maa2c.n_steps += 1
    
    maa2c.memory.push(states, actions, rewards)
    
@gen.coroutine
def main():
    env = DiplomacyEnv()
#     env.seed(RANDOM_SEED)
    env_eval = DiplomacyEnv()
#     env_eval.seed(RANDOM_SEED)
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

    episodes =[]
    eval_rewards =[]
    while maa2c.n_episodes < MAX_EPISODES:
        interact(env, maa2c)
        if maa2c.n_episodes >= EPISODES_BEFORE_TRAIN:
            maa2c.train()
        if maa2c.episode_done and ((maa2c.n_episodes+1)%EVAL_INTERVAL == 0):
            rewards, _ = maa2c.evaluation(env_eval, EVAL_EPISODES)
            rewards_mu, rewards_std = ma_agg_double_list(rewards)
            for agent_id in range (N_AGENTS):
                print("Episode %d, Agent %d, Average Reward %.2f" % (maa2c.n_episodes+1, agent_id, rewards_mu[agent_id]))
            episodes.append(maa2c.n_episodes+1)
            eval_rewards.append(rewards_mu)
  
    stop_io_loop()
        
if __name__ == '__main__':
  start_io_loop(main)
