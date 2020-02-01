# noinspection PyUnresolvedReferences
from constants import THIS_DIR
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import re


def loop(in_dir):
    sub_dirs = \
        [os.path.basename(f.path) for f in os.scandir(in_dir) if f.is_dir()]

    for this_d in sub_dirs:
        main(os.path.join(THIS_DIR, in_dir, this_d))


def _get_rewards_actions(df):
    return df[['episode', 'reward']].groupby('episode').sum(), \
           df[['episode', 'action_taken']].groupby('episode').count()


def plot_training(df, exp_dir):
    rewards, actions = _get_rewards_actions(df)

    _plot_helper_train(s_in=rewards, save_dir=exp_dir, term='Reward',
                       window=True)
    _plot_helper_train(s_in=actions, save_dir=exp_dir, term='Action',
                       window=True)


def _plot_helper_train(s_in, save_dir, term, window=True):
    l_term = term.lower()
    # Plot all points.
    fig1, ax1 = plt.subplots()
    ax1.scatter(x=np.arange(len(s_in)), y=s_in.to_numpy())
    ax1.set_title(f'Total Training Episode {term}s per Episode')
    ax1.set_xlabel('Episode Number')
    ax1.set_ylabel(f'Total Episode {term}s')
    ax1.grid(True, 'both')
    plt.tight_layout()

    fig1.savefig(os.path.join(save_dir, f'episode_{l_term}.png'), format='png')
    fig1.savefig(os.path.join(save_dir, f'episode_{l_term}.eps'), format='eps')
    plt.close(fig1)

    if window:
        # Do a 100 episode rolling average.
        rolling = s_in.rolling(100).mean().to_numpy()
        # Get rid of the NaNs at the beginning.
        rolling = rolling[~pd.isna(rolling)]
        assert len(rolling) == len(s_in) - 99
        fig2, ax2 = plt.subplots()
        ax2.scatter(x=np.arange(len(rolling)) + 99, y=rolling)
        ax2.set_xlabel('Episode Number')
        ax2.set_ylabel(f'Average Total Episode {term}s')
        ax2.set_title(f'100 Episode Average Episode {term}s (Sliding Window)')
        ax2.grid(True, 'both')
        plt.tight_layout()

        fig2.savefig(
            os.path.join(save_dir, f'average_{l_term}.png'), format='png')
        fig2.savefig(
            os.path.join(save_dir, f'average_{l_term}.eps'), format='eps')
        plt.close(fig2)


def plot_testing(df, exp_dir):
    rewards, actions = _get_rewards_actions(df)
    reward_s = rewards['reward']
    action_s = actions['action_taken']
    _plot_helper_test(s_in=reward_s, save_dir=exp_dir, term='Reward')
    _plot_helper_test(s_in=action_s, save_dir=exp_dir,
                      term='Action Count')


def _plot_helper_test(s_in, save_dir, term):
    l_term = term.lower().replace(' ', '_')
    # Get value counts.
    v_counts = s_in.value_counts(normalize=True).sort_index(
        ascending=('action' not in l_term)) * 100
    # Round.
    v_counts.index = np.round(v_counts.index).to_numpy(dtype=int)
    # Create bar chart.
    fig = plt.figure()
    ax = v_counts.plot(kind='bar')
    # ax.bar(v_counts.index, v_counts.to_numpy(), align='center')
    ax.set_xlabel(f'{term}s')
    ax.set_ylabel(f'Percentage of Episodes with Given {term}')
    ax.grid(True, which='major', axis='y')
    ax.set_title(f'Normalized Frequency of Testing {term}s')
    # Use nifty stack overflow answer to add labels.
    add_value_labels(ax)
    ax.set_axisbelow(True)
    # plt.tight_layout()
    # Save.
    # fig = plt.gcf()
    fig.savefig(os.path.join(save_dir, f'test_{l_term}.png'), format='png')
    fig.savefig(os.path.join(save_dir, f'test_{l_term}.eps'), format='eps')
    plt.close(fig)


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.

    Source: https://stackoverflow.com/a/48372659/11052174
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with two decimal places
        label = "{:.2f}".format(y_value)

        # Create annotation
        ax.annotate(
            # Use `label` as label
            label,
            # Place label at end of the bar
            (x_value, y_value),
            # Place label at end of the bar
            xytext=(0, space),
            # Interpret `xytext` as offset in points
            textcoords="offset points",
            # Horizontally center label
            ha='center',
            # Vertically align label differently for positive and
            # negative values.
            va=va)


def main(run_dir):
    print('*' * 80)
    print(f'RUN DIRECTORY: {run_dir}')
    print('')

    # Load training data.
    df_train = pd.read_csv(os.path.join(run_dir, 'log_train.csv'))

    # Plot training rewards.
    plot_training(df=df_train, exp_dir=run_dir)

    # Load testing data.
    df_test = pd.read_csv(os.path.join(run_dir, 'log_test.csv'))

    # Plot testing data.
    plot_testing(df=df_test, exp_dir=run_dir)

    # Read the info file.
    with open(os.path.join(run_dir, 'info.txt'), 'r') as f:
        s = f.read()

    # Extract the run time.
    m = re.match('(?:Training took\s)(.+)(?:\sseconds.)', s)
    train_time = float(m.group(1))
    print(f'Training took {train_time}')

    # Count actions taken per episode in testing.
    test_episode_actions = \
        df_test[['episode', 'reward']].groupby('episode').count()

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


if __name__ == '__main__':
    for d in ['gm_contingency_512_512', 'gm_contingency_128_256',
              'gm_contingency_64_128_512', 'gm_contingency_64_128',
              'gm_contingency_32_64_long', 'gm_contingency_32_32',
              'gm_contingency_64_64_long', 'gm_contingency_128_128',
              'gm_contingency_128_128_super_long']:
        loop(d)
