# This is a newly implemented script.
# It is used to generate evaluation plots based on our experiment records.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv_files(exps):
    success_rates = []
    scores = []
    for e in exps:
        csv_path = os.path.join("exp", e, "logs", "log.csv")
        df = pd.read_csv(csv_path)
        sr = df['success_rate']
        sc = df['eval_score']
        success_rates.append(sr)
        scores.append(sc)
    return success_rates, scores

def plot_success(success_rates, legends, colors, ax):
    assert len(success_rates) == len(legends)
    for i in range(len(success_rates)):
        s = success_rates[i]
        ss = smooth(s)
        ax.plot(np.arange(len(s)), s, alpha=0.25, color=colors[i])
        ax.plot(np.arange(len(ss)), ss, color=colors[i], label=legends[i])
    ax.set_xlabel("iterations", fontsize=15)
    ax.set_ylabel("(smoothed) success rate", fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=15)
    #plt.show()

def plot_score(scores, legends, colors, ax):
    assert len(success_rates) == len(legends)
    for i in range(len(success_rates)):
        s = scores[i]
        ss = smooth(s)
        ax.plot(np.arange(len(s)), s, alpha=0.25, color=colors[i])
        ax.plot(np.arange(len(ss)), ss, color=colors[i], label=legends[i])
    ax.set_xlabel("iterations", fontsize=15)
    ax.set_ylabel("(smoothed) score", fontsize=15)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=15)
    #plt.show()

def smooth(array, interval=5):
    smoothed = []
    for i in range(len(array)):
        smoothed.append(np.mean(array[i:i+interval]))
    return np.array(smoothed)
    

if __name__ == "__main__":

    #exps = ["dapg_exp", "bcrl_exp", "reinforce_exp", "rl_scratch_exp"]
    #legends = ["DAPG", "NPG", "REINFORCE", "Learning from scratch"]

    #exps = ["dapg_exp", "dagger_exp"]
    #legends = ["DAPG", "DAPG (off-policy)"]

    #exps = ["dapg_exp", "actor_critic_exp_pretrain", "dapg_gae_opt_exp"]
    #legends = ["DAPG", "Actor-Critic", "DAPG (with GAE)"]

    #exps = ["dapg_exp", "dapg_lstm_exp"]
    #legends = ["DAPG", "DAPG (with LSTM)"]

    #exps = ["dapg_exp", "dapg_gae_opt_exp", "dapg_lstm_exp", "dapg_lstm_gae_exp"]
    #legends = ["DAPG", "DAPG (with GAE)", "DAPG (with LSTM)", "DAPG (with GAE + LSTM)"]

    exps = ["actor_critic_exp", "rl_dapg_gae_01", "rl_dapg_gae_03", "rl_dapg_gae_050", "rl_dapg_gae_080",  \
            "dapg_gae_opt_exp", "dapg_gae_exp"]
    legends = [r'$\lambda=0$', r'$\lambda=0.1$', r'$\lambda=0.3$', r'$\lambda=0.5$', r'$\lambda=0.8$', \
               r'$\lambda=0.97$', r'$\lambda=1.0$']
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    success_rates, scores = read_csv_files(exps)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_success(success_rates, legends, colors, axes[0])
    plot_score(scores, legends, colors, axes[1])
    plt.tight_layout()
    plt.show()
