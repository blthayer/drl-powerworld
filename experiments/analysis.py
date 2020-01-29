# noinspection PyUnresolvedReferences
from constants import THIS_DIR
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    df_train = pd.read_csv(
        os.path.join(THIS_DIR, 'gridmind_reproduce', 'log_train.csv'))
    df_test = pd.read_csv(
        os.path.join(THIS_DIR, 'gridmind_reproduce', 'log_test.csv'))

    # Let's compute and plot average rewards over training.
    episode_rewards = df_train[['episode', 'reward']].groupby('episode').sum()

    # Plot all episode rewards.
    fig1, ax1 = plt.subplots()
    ax1.scatter(x=np.arange(len(episode_rewards)), y=episode_rewards.to_numpy())
    ax1.set_title('Total Training Episode Rewards')
    ax1.set_xlabel('Episode Number')
    ax1.set_ylabel('Total Episode Reward')

    # Do a rolling average.
    rolling = episode_rewards.rolling(100).mean().to_numpy()
    # Get rid of the NaNs at the beginning.
    rolling = rolling[~pd.isna(rolling)]
    assert len(rolling) == len(episode_rewards) - 99
    fig2, ax2 = plt.subplots()
    ax2.scatter(x=np.arange(len(rolling)) + 99, y=rolling)
    # Hard-code the reward cap of 200.
    # ax2.set_ylim([rolling.min(), 200])
    ax2.set_xlabel('Episode Number')
    ax2.set_ylabel('100 Ep. Avg. Total Ep. Reward')
    ax2.set_title('100 Episode Sliding Window Average Episode Rewards')

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

    pass

    # "line" graph for action_taken.
    # Relatively useless
    # plt.plot(df['action_taken'])

    # # "bar" graph of all actions
    # action_count = df_test['action_taken'].value_counts(
    #     normalize=False, sort=False, ascending=False, dropna=True)
    # action_count_norm = df_test['action_taken'].value_counts(
    #     normalize=True, sort=True, ascending=False, dropna=True
    # )
    # # Still have to sort it? Fine.
    # action_count.sort_index(inplace=True)
    # # Bar graph.
    # #
    # plt.bar(x=action_count.index, height=action_count.to_numpy())
    plt.show()


if __name__ == '__main__':
    main()
