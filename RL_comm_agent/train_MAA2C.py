from pytorch_DRL.MAA2C import MAA2C
from pytorch_DRL.common.utils import ma_agg_double_list
from DiplomacyEnv import DiplomacyEnv

import sys
import numpy as np
import matplotlib.pyplot as plt
import random

# MAA2C: https://github.com/ChenglongChen/pytorch-DRL/

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


def interact(env, maa2c):
    dip_step = 0
    if (maa2c.max_steps is not None) and (maa2c.n_steps >= maa2c.max_steps):
        # env_state is dictionary
        maa2c.env_state = self.env.reset()
        # tranfrom from dict to arr
        maa2c.env_state = maa2c.agentdict_to_arr(maa2c.env_state)
    dip_game = env.dip_game
    dip_player = env.dip_player
    dip_player.init_communication(dip_game.game.powers)
    while not dip_game.game.is_game_done and dip_step < maa2c.roll_out_n_steps:
        
        for sender in dip_game.powers:
            for recipient in dip_game.powers:
                if sender != recipient and not dip_game.powers[sender].is_eliminated() and not dip_game.powers[recipient].is_eliminated():
                    orders = yield {power_name: dip_player.get_orders(dip_game.game, power_name) for power_name in dip_game.powers}
                    stance = dip_player.stance[sender][recipient] 
                    env.set_power_state(sender, dip_player, stance)
                    maa2c.env_state = env.cur_obs
                    action = maa2c.exploration_action(maa2c.env_state)
                    action_dict = {agent_id: action[agent_id] for agent_id in range(self.n_agents)}
                    env.step(action_dict)

        orders = yield {power_name: dip_player.get_orders(dip_game.game, power_name) for power_name in dip_game.powers}
        for power_name, power_orders in orders.items():
           dip_game.game.set_orders(power_name, power_orders)
        dip_game.game_process()
        dip_step +=1
    
    
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
        interact()
        if maa2c.n_episodes >= EPISODES_BEFORE_TRAIN:
            maa2c.train()
        if maa2c.episode_done and ((maa2c.n_episodes+1)%EVAL_INTERVAL == 0):
            rewards, _ = maa2c.evaluation(env_eval, EVAL_EPISODES)
            rewards_mu, rewards_std = ma_agg_double_list(rewards)
            for agent_id in range (N_AGENTS):
                print("Episode %d, Agent %d, Average Reward %.2f" % (maa2c.n_episodes+1, agent_id, rewards_mu[agent_id]))
            episodes.append(maa2c.n_episodes+1)
            eval_rewards.append(rewards_mu)
  

        
if __name__ == '__main__':
  main()
