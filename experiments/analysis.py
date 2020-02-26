# noinspection PyUnresolvedReferences,PyPackageRequirements
from constants import THIS_DIR, DATA_DIR, TEST_EPISODES, add_value_labels
from gym_powerworld.envs.voltage_control_env import V_TOL
import pandas as pd
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re

# Set up some default plotting.
# 3" x 3" should work well for creating 2 x 2 sub plots.
# mpl.rcParams['figure.figsize'] = [3, 3]
# Use 300 DPI
mpl.rcParams['savefig.dpi'] = 300
# Use tight bbox for saving.
# mpl.rcParams['savefig.bbox'] = 'tight'
# No padding.
# mpl.rcParams['savefig.pad_inches'] = 0.0


def loop(in_dir):
    sub_dirs = \
        [os.path.basename(f.path) for f in os.scandir(in_dir) if f.is_dir()]

    pct_success_arr = np.zeros(len(sub_dirs))
    oob_success_arr = np.zeros_like(pct_success_arr)
    mean_reward_arr = np.zeros_like(pct_success_arr)
    num_test_actions_arr = np.zeros_like(pct_success_arr)
    time_arr = np.zeros_like(pct_success_arr)
    oob_arr = np.zeros_like(pct_success_arr)

    for i, this_d in enumerate(sub_dirs):
        pct_success, oob_success, mean_reward, num_test_actions, train_time,\
            pct_oob = main(os.path.join(DATA_DIR, in_dir, this_d))

        pct_success_arr[i] = pct_success
        oob_success_arr[i] = oob_success
        mean_reward_arr[i] = mean_reward
        num_test_actions_arr[i] = num_test_actions
        time_arr[i] = train_time
        oob_arr[i] = pct_oob

    df = pd.DataFrame({'Success %': pct_success_arr,
                       'Success % O.O.B.': oob_success_arr,
                       'Mean Reward': mean_reward_arr,
                       'Percentage of Episodes Start OOB': oob_arr,
                       'Num Test Actions': num_test_actions_arr,
                       'Training time': time_arr})
    df.to_csv(os.path.join(in_dir, 'summary.csv'))

    # It's okay to take means of means since all our tests have the
    # same number of samples.
    return pct_success_arr.mean(), oob_success_arr.mean(), \
        mean_reward_arr.mean(), num_test_actions_arr.mean(), time_arr.mean()


def _get_rewards_actions(df):
    return df[['episode', 'reward']].groupby('episode').sum(), \
           df[['episode', 'action_taken']].groupby('episode').count()


def plot_training(df, exp_dir):
    rewards, actions = _get_rewards_actions(df)

    _plot_helper_train(s_in=rewards, save_dir=exp_dir, term='Reward')
    _plot_helper_train(s_in=actions, save_dir=exp_dir, term='Action')

    # Get series of successes.
    success, start_oob = _get_success_series(df)

    # Get the rolling 100 episode average.
    r = success.rolling(100).mean() * 100

    # Get rid of the NaNs at the beginning.
    r = r[~pd.isna(r)]

    fig, ax = plt.subplots()
    ax.scatter(x=np.arange(len(r)) + 99, y=r)
    ax.set_xlabel('Episode Number')
    ax.set_ylabel(f'Average Success Percentage')
    ax.set_title(f'100 Episode Average Success Percentage (Sliding Window)')
    ax.grid(True, 'both')
    ax.set_axisbelow(True)
    plt.tight_layout()

    fig.savefig(
        os.path.join(exp_dir, f'train_success.png'), format='png')
    fig.savefig(
        os.path.join(exp_dir, f'train_success.eps'), format='eps')
    plt.close(fig)

    return rewards, actions


def _plot_helper_train(s_in, save_dir, term):
    l_term = term.lower()

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
    ax2.set_axisbelow(True)
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
    _test_reward_hist(s_in=reward_s, save_dir=exp_dir)
    _test_action_hist(s_in=action_s, save_dir=exp_dir)
    # _plot_helper_test(s_in=reward_s, save_dir=exp_dir, term='Reward')
    _plot_helper_test(s_in=action_s, save_dir=exp_dir,
                      term='Action Count')

    return rewards, actions


# noinspection DuplicatedCode
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
    # Ensure the ylim is 0 to 100.
    ax.set_ybound(0.0, 100.0)
    # ax.bar(v_counts.index, v_counts.to_numpy(), align='center')
    ax.set_xlabel(f'{term}s')
    ax.set_ylabel(f'Percentage of Episodes with Given {term}')
    ax.grid(True, which='major', axis='y')
    ax.set_title(f'Normalized Frequency of Testing {term}s')
    # Use nifty stack overflow answer to add labels.
    add_value_labels(ax)
    ax.set_axisbelow(True)
    plt.tight_layout()
    # Save.
    # fig = plt.gcf()
    fig.savefig(os.path.join(save_dir, f'test_{l_term}.png'), format='png')
    fig.savefig(os.path.join(save_dir, f'test_{l_term}.eps'), format='eps')
    plt.close(fig)


# noinspection DuplicatedCode
def _test_reward_hist(s_in, save_dir):
    # Initialize figure.
    fig, ax = plt.subplots()
    # Create histogram.
    counts, bins, patches = ax.hist(s_in.to_numpy())
    # Normalize. https://stackoverflow.com/a/22241514/11052174
    for item in patches:
        item.set_height(item.get_height()/s_in.shape[0] * 100)
    ax.set_ybound(0.0, 100.0)
    ax.set_xlabel('Episode Reward Bins')
    ax.set_ylabel(f'Percentage of Rewards Falling in Reward Bin')
    ax.set_title('Normalized Histogram of Testing Episode Rewards')
    add_value_labels(ax)
    ax.grid(True, which='major', axis='y')
    ax.set_xticks(bins)
    ax.xaxis.set_tick_params(rotation=45)
    plt.tight_layout()
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(save_dir, 'test_reward_hist.png'), format='png')
    fig.savefig(os.path.join(save_dir, 'test_reward_hist.eps'), format='eps')
    plt.close(fig)
    pass


# noinspection DuplicatedCode
def _test_action_hist(s_in, save_dir):
    # Initialize figure.
    fig, ax = plt.subplots()
    # Create histogram.
    counts, bins, patches = ax.hist(s_in.to_numpy())
    # Normalize. https://stackoverflow.com/a/22241514/11052174
    for item in patches:
        item.set_height(item.get_height() / s_in.shape[0] * 100)
    ax.set_ybound(0.0, 100.0)
    ax.set_xlabel('Episode Action Count Bins')
    ax.set_ylabel(f'Percentage of Episodes Falling in the Action Count Bin')
    ax.set_title('Normalized Histogram of Testing Episode Action Counts')
    add_value_labels(ax)
    ax.grid(True, which='major', axis='y')
    ax.set_xticks(bins)
    ax.xaxis.set_tick_params(rotation=45)
    plt.tight_layout()
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(save_dir, 'test_action_hist.png'), format='png')
    fig.savefig(os.path.join(save_dir, 'test_action_hist.eps'), format='eps')
    plt.close(fig)


def _get_success_series(df_in):
    # Get array of NaNs in the "reward" column. This will notate
    # episode start.
    ep_start = df_in['reward'].isna().to_numpy()

    # Roll the array backwards so we get an index into the end of
    # the episode.
    ep_end = np.roll(ep_start, -1)

    # Extract voltage columns.
    v_col = df_in.columns[df_in.columns.str.startswith('bus_')]

    # Extract voltage data for the end of each episode.
    v_test_end = df_in.loc[ep_end, v_col]

    # Extract voltage data for the start of each episode.
    v_test_start = df_in.loc[ep_start, v_col]

    # Check to see if all voltages are in bounds. This indicates
    # success.
    low_end = v_test_end < (0.95 - V_TOL)
    high_end = v_test_end > (1.05 + V_TOL)
    low_start = v_test_start < (0.95 - V_TOL)
    high_start = v_test_start > (1.05 + V_TOL)

    # noinspection PyUnresolvedReferences
    success_series = ((~low_end) & (~high_end)).all(axis=1)
    start_oob = (low_start | high_start).any(axis=1)

    return success_series, start_oob


def main(run_dir):
    print('*' * 80)
    print(f'RUN DIRECTORY: {run_dir}')
    print('')

    # Load training data.
    try:
        df_train = pd.read_csv(os.path.join(run_dir, 'log_train.csv'))
    except FileNotFoundError:
        pass
    else:
        # Plot training rewards.
        train_rewards, train_actions = \
            plot_training(df=df_train, exp_dir=run_dir)

    # Load testing data.
    df_test = pd.read_csv(os.path.join(run_dir, 'log_test.csv'))

    # Plot testing data.
    test_rewards, test_actions = plot_testing(df=df_test, exp_dir=run_dir)

    # Read the info file.
    try:
        with open(os.path.join(run_dir, 'info.txt'), 'r') as f:
            s = f.read()
    except FileNotFoundError:
        train_time = np.nan
    else:
        # Extract the run time.
        m = re.match('(?:Training took\s)(.+)(?:\sseconds.)', s)
        train_time = float(m.group(1))
        print(f'Training took {train_time}')

    # Get series related to the testing success.
    success_series, start_oob_series = _get_success_series(df_test)
    # Compute overall testing success.
    pct_success = success_series.sum() / success_series.shape[0]
    # Compute success for episodes which started out of bounds
    oob_success = (success_series[start_oob_series.to_numpy()].sum()
                   / start_oob_series.sum())
    # Compute percentage of episodes which started out of bounds.
    pct_ep_oob = start_oob_series.to_numpy().sum() / success_series.shape[0]
    assert success_series.shape[0] == TEST_EPISODES
    assert start_oob_series.shape[0] == TEST_EPISODES

    # # Count actions taken per episode in testing.
    # test_episode_actions = \
    #     df_test[['episode', 'reward']].groupby('episode').count()

    # Get the unique actions taken in testing.
    test_actions = df_test['action_taken'].unique()
    print(f'Actions taken during testing: {test_actions}')

    test_ep_rewards = df_test[['episode', 'reward']].groupby('episode').sum()
    print('Description of testing episode rewards:')
    print(test_ep_rewards.describe())

    # # Let's examine voltage issues.
    # # We want the NaN action_taken values, because they denote the start
    # # of an episode.
    # ep_start = df_test.loc[pd.isna(df_test['action_taken']), :]
    # test_v = \
    #     ep_start.loc[:,
    #         ep_start.columns[ep_start.columns.str.startswith('bus_')]].round(6)
    #
    # # assert test_v.shape == (TEST_EPISODES, 14)
    #
    # under_voltage = (test_v < 0.95).sum(axis=1)
    # over_voltage = (test_v > 1.05).sum(axis=1)
    #
    # print('Description of under voltages in testing set:')
    # print(under_voltage.describe())
    # print('Description of over voltages in testing set:')
    # print(over_voltage.describe())

    # Return percent success, mean reward, and num unique testing
    # actions. Subtract one since we have np.nan in there for actions.
    return pct_success, oob_success, test_rewards.mean()[0],\
        test_actions.shape[0] - 1, train_time, pct_ep_oob


if __name__ == '__main__':
    # Directories to loop over.
    dir_list = \
        [os.path.basename(f.path) for f in os.scandir(DATA_DIR) if f.is_dir()]
    # dir_list = ['gm_con_512_1024', 'gm_con_512_512', 'gm_con_256_512',
    #             'gm_con_256_256', 'gm_con_128_256', 'gm_con_128_128',
    #             'gm_con_64_128', 'gm_con_64_64', 'gm_con_32_64',
    #             'gm_con_32_32', 'gm_64_64']
    #             # 'gm_contingency_128_128_super_long']
    # dir_list = ['de_hard_branch_and_gen_state_64_128',]

    dir_list = ['de_hard_mod_gen_state_64_128_long',
                'de_hard_mod_gen_state_512_1024_long',
                'de_hard_mod_branch_and_gen_state_64_128']
    exclude_list = []

    # Initialize DataFrame to hold summary stats for each run.
    df_ = pd.DataFrame(np.zeros((len(dir_list), 5)),
                       columns=['run', 'pct_success', 'pct_success_oob',
                                'mean_reward', 'mean_num_actions'])

    df_['run'] = ''

    for i_, d_ in enumerate(dir_list):
        if d_.startswith('_'):
            continue
        if d_ in exclude_list:
            continue
        s_, s_oob, r_, a_, t_ = loop(os.path.join(DATA_DIR, d_))
        df_.loc[i_, 'run'] = d_
        df_.loc[i_, 'pct_success'] = s_
        df_.loc[i_, 'pct_success_oob'] = s_oob
        df_.loc[i_, 'mean_reward'] = r_
        df_.loc[i_, 'mean_num_actions'] = a_
        df_.loc[i_, 'mean_train_time'] = t_

    df_.to_csv('summary_stats.csv')

