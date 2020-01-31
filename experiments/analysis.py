# noinspection PyUnresolvedReferences
from constants import THIS_DIR
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import re


def loop():
    num_runs = 10
    num_episodes = np.zeros(num_runs)
    train_time = np.zeros(num_runs)
    aced = np.zeros(num_runs)

    for i in range(10):
        ep, t, ace = main(i)
        num_episodes[i] = ep
        train_time[i] = t
        aced[i] = ace

    print('*' * 80)
    print('Overall statistics:')
    print(f'Mean training time: {train_time.mean()}')
    print(f'Mean number of training episodes: {num_episodes.mean()}')
    print(f'Mean testing episodes "aced": {aced.mean()}')

    fig, ax = plt.subplots()
    ax.boxplot()


def main(run_num):
    print('*' * 80)
    print(f'RUN {run_num}')
    print('')

    base_dir = os.path.join(THIS_DIR, 'gridmind_reproduce')
    run_dir = os.path.join(base_dir, f'run_{run_num}')

    df_train = pd.read_csv(os.path.join(run_dir, 'log_train.csv'))
    df_test = pd.read_csv(os.path.join(run_dir, 'log_test.csv'))

    # Read the info file.
    with open(os.path.join(run_dir, 'info.txt'), 'r') as f:
        s = f.read()

    # Extract the run time.
    m = re.match('(?:Training took\s)(.+)(?:\sseconds.)', s)
    train_time = float(m.group(1))

    # Let's compute and plot average rewards over training.
    train_episode_rewards = \
        df_train[['episode', 'reward']].groupby('episode').sum()
    test_episode_actions = \
        df_test[['episode', 'reward']].groupby('episode').count()

    # Plot all episode rewards.
    fig1, ax1 = plt.subplots()
    ax1.scatter(x=np.arange(len(train_episode_rewards)),
                y=train_episode_rewards.to_numpy())
    ax1.set_title('Total Training Episode Rewards')
    ax1.set_xlabel('Episode Number')
    ax1.set_ylabel('Total Episode Reward')
    ax1.grid(True, 'both')
    plt.tight_layout()

    fig1.savefig(os.path.join(run_dir, 'episode_rewards.png'), format='png')
    fig1.savefig(os.path.join(run_dir, 'episode_rewards.eps'), format='eps')

    # Do a rolling average.
    rolling = train_episode_rewards.rolling(100).mean().to_numpy()
    # Get rid of the NaNs at the beginning.
    rolling = rolling[~pd.isna(rolling)]
    assert len(rolling) == len(train_episode_rewards) - 99
    fig2, ax2 = plt.subplots()
    ax2.scatter(x=np.arange(len(rolling)) + 99, y=rolling)
    # Hard-code the reward cap of 200.
    # ax2.set_ylim([rolling.min(), 200])
    ax2.set_xlabel('Episode Number')
    ax2.set_ylabel('Average Total Episode Reward')
    ax2.set_title('100 Episode Average Episode Rewards (Sliding Window)')
    ax2.grid(True, 'both')
    plt.tight_layout()

    fig2.savefig(os.path.join(run_dir, 'average_rewards.png'), format='png')
    fig2.savefig(os.path.join(run_dir, 'average_rewards.eps'), format='eps')

    # Get the unique actions taken in testing.
    test_actions = df_test['action_taken'].unique()
    print(f'Actions taken during testing: {test_actions}')

    test_ep_rewards = df_test[['episode', 'reward']].groupby('episode').sum()
    print('Description of testing episode rewards:')
    print(test_ep_rewards.describe())

    # Let's examine voltage issues.
    # We want the NaN action_taken values, because they denote the start
    # of an episode.
    ep_start = df_test.loc[pd.isna(df_test['action_taken']), :]
    test_v = \
        ep_start.loc[:,
        ep_start.columns[ep_start.columns.str.startswith('bus_')]].round(6)

    assert test_v.shape == (2000, 14)

    under_voltage = (test_v < 0.95).sum(axis=1)
    over_voltage = (test_v > 1.05).sum(axis=1)

    print('Description of under voltages in testing set:')
    print(under_voltage.describe())
    print('Description of over voltages in testing set:')
    print(over_voltage.describe())

    # Return the number of training episodes, training time, and
    # number of "aced" test episodes.
    return train_episode_rewards.shape[0], train_time,\
        (test_ep_rewards == 200.0).sum()


if __name__ == '__main__':
    loop()
