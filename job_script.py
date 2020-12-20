# This is a modified script originally located at:
# https://github.com/aravindr93/hand_dapg/blob/master/dapg/examples/job_script.py
# The modifications mainly involve adding extra arguments and instance declarations
# to allow for running experiments with our newly implemented algorithms.


"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import os
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse

from mjrl.algos.batch_reinforce import BatchREINFORCE
from mjrl.algos.dagger import DAgger
from mjrl.algos.actor_critic import ActorCritic
from mjrl.algos.dapg_lstm import DAPGlstm

from mjrl.policies.gaussian_lstm import LSTM

from mjrl.algos.behavior_cloning_lstm import BClstm

import copy


if __name__ == "__main__":
    # ===============================================================================
    # Get command line arguments
    # ===============================================================================
    
    parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data.')
    parser.add_argument('--output', type=str, required=True, help='location to store results')
    parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
    args = parser.parse_args()
    JOB_DIR = args.output
    if not os.path.exists(JOB_DIR):
        os.mkdir(JOB_DIR)
    with open(args.config, 'r') as f:
        job_data = eval(f.read())
    assert 'algorithm' in job_data.keys()
    # assert any([job_data['algorithm'] == a for a in ['NPG', 'BCRL', 'DAPG']])
    job_data['lam_0'] = 0.0 if 'lam_0' not in job_data.keys() else job_data['lam_0']
    job_data['lam_1'] = 0.0 if 'lam_1' not in job_data.keys() else job_data['lam_1']
    EXP_FILE = JOB_DIR + '/job_config.json'
    with open(EXP_FILE, 'w') as f:
        json.dump(job_data, f, indent=4)
    
    # ===============================================================================
    # Train Loop
    # ===============================================================================

    e = GymEnv(job_data['env'])
    if job_data['algorithm'] == 'DAPGLSTM':
        policy = LSTM(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    else:
        policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                       epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])
    
    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
        if job_data['algorithm'] != 'DAPGLSTM':
            bc_agent = BC(demo_paths, policy=policy, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                          lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        else:
            bc_agent = BClstm(demo_paths, policy=policy, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                              lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)
    
        ts = timer.time()
        print("========================================")
        print("Running BC with expert demonstrations")
        print("========================================")
        bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        if job_data['eval_rollouts'] >= 1:
            if job_data['algorithm'] == 'DAPGLSTM':
                score = e.evaluate_policy_lstm(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
            else:
                score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
            print("Score with behavior cloning = %f" % score[0][0])

    if job_data['algorithm'] == 'NGP' or \
       job_data['algorithm'] == 'BCRL' or \
       job_data['algorithm'] == 'REINFORCE' or \
       job_data['algorithm'] == 'DAGGER':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None
        print("No demo augmented.")
    else:
        print("With demo augmented.")
    
    # ===============================================================================
    # RL Loop
    # ===============================================================================

    if job_data['algorithm'] == 'REINFORCE':
        rl_agent = BatchREINFORCE(e, policy, baseline,
                                  learning_rate=job_data['rl_step_size'],
                                  seed=job_data['seed'], save_logs=True)
    elif job_data['algorithm'] == 'DAGGER':
        #expert_policy = pickle.load(open("../policies/pen-v0.pickle", 'rb'))
        #expert_policy = pickle.load(open("./dapg_exp/iterations/best_policy.pickle", 'rb'))
        #expert_policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
        expert_policy = copy.deepcopy(policy)
        rl_agent = DAgger(e, policy, expert_policy, baseline, demo_paths,
                          normalized_step_size=job_data['rl_step_size'],
                          lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
                          seed=job_data['seed'], save_logs=True)
    elif job_data['algorithm'] == 'ACTOR_CRITIC':
        rl_agent = ActorCritic(e, policy, baseline, demo_paths,
                               normalized_step_size=job_data['rl_step_size'],
                               lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
                               seed=job_data['seed'], save_logs=True
                               )
    elif job_data['algorithm'] == 'DAPGLSTM':
        rl_agent = DAPGlstm(e, policy, baseline, demo_paths,
                            normalized_step_size=job_data['rl_step_size'],
                            lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
                            seed=job_data['seed'], save_logs=True
                            )
    else:
        rl_agent = DAPG(e, policy, baseline, demo_paths,
                        normalized_step_size=job_data['rl_step_size'],
                        lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
                        seed=job_data['seed'], save_logs=True
                        )
    
    print("========================================")
    print("Starting reinforcement learning phase")
    print("========================================")
    
    ts = timer.time()
    train_agent(job_name=JOB_DIR,
                agent=rl_agent,
                seed=job_data['seed'],
                niter=job_data['rl_num_iter'],
                gamma=job_data['rl_gamma'],
                gae_lambda=job_data['rl_gae'],
                num_cpu=job_data['num_cpu'],
                sample_mode='trajectories',
                num_traj=job_data['rl_num_traj'],
                save_freq=job_data['save_freq'],
                evaluation_rollouts=job_data['eval_rollouts'],
                lstm=True if (job_data['algorithm'] == 'DAPGLSTM') else False)
    print("time taken = %f" % (timer.time()-ts))
