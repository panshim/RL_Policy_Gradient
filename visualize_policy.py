# This is a modified script originally located at:
# https://github.com/aravindr93/hand_dapg/blob/master/dapg/utils/visualize_policy.py
# The modification is merely adding a command for visualizing the newly added lstm policy

import mj_envs
import click 
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy --env_name relocate-v0 --policy policies/relocate-v0.pickle --mode evaluation\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', required=True)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--lstm', type=bool, default=False)
def main(env_name, policy, mode, lstm):
    e = GymEnv(env_name)
    pi = pickle.load(open(policy, 'rb'))
    # render policy
    if lstm:
        e.visualize_policy_lstm(pi, num_episodes=100, horizon=e.horizon, mode=mode)
    else:
        e.visualize_policy(pi, num_episodes=100, horizon=e.horizon, mode=mode)

if __name__ == '__main__':
    main()
