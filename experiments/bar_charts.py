"""Read the overall summary file and create bar charts grouped by
observations given to the agent.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# noinspection PyUnresolvedReferences,PyPackageRequirements
from constants import add_value_labels
mpl.rcParams['savefig.dpi'] = 300

# Hard code the 'codes' used for each group
GROUP_LEN = 5
V_ONLY_GROUP = ['de_64_64', 'de_mod_64_64', 'de_v_scaled_64_64',
                'de_v_scaled_mod_64_64', 'de_v_scaled_clipped_r_mod_64_64']
GEN_OBS_GROUP = ['de_gen_state_64_64', 'de_gen_state_mod_64_64',
                 'de_gen_state_v_scaled_64_64',
                 'de_gen_state_v_scaled_mod_64_64',
                 'de_gen_state_v_scaled_clipped_r_mod_64_64']
BRANCH_OBS_GROUP = ['de_branch_state_64_64', 'de_branch_state_mod_64_64',
                    'de_branch_state_v_scaled_64_64',
                    'de_branch_state_v_scaled_mod_64_64',
                    'de_branch_state_v_scaled_clipped_r_mod_64_64']
BRANCH_AND_GEN_GROUP = ['de_branch_and_gen_state_64_64',
                        'de_branch_and_gen_state_mod_64_64',
                        'de_branch_and_gen_state_v_scaled_64_64',
                        'de_branch_and_gen_state_v_scaled_mod_64_64',
                        'de_branch_and_gen_state_v_scaled_clipped_r_mod_64_64']
ALL_GROUPS = [V_ONLY_GROUP, GEN_OBS_GROUP, BRANCH_OBS_GROUP,
              BRANCH_AND_GEN_GROUP]
GROUP_STRINGS = ['Bus Voltage Only', 'Bus Voltage and Generator State',
                 'Bus Voltage and Branch State',
                 'Bus Voltage, Generator State, and Branch State']
SUBGROUP_STRINGS = [
    'Per Unit Voltage, Reward Scheme 1',
    'Per Unit Voltage, Reward Scheme 1, Unique-Actions-Per-Episode',
    'Min-Max Scaled Voltage, Reward Scheme 1',
    'Min-Max Scaled Voltage, Reward Scheme 1, Unique-Actions-Per-Episode',
    'Min-Max Scaled Voltage, Reward Scheme 2, Unique-Actions-Per-Episode'
]

# Hard code the graph-based and random agent percent successes O.O.B.
RANDOM_SUCCESS_OOB = 0.1652
GRAPH_SUCCESS_OOB = 0.4079


def main():
    # Read the file. Note it's been renamed from the original output
    # from analysis.py and only contains 14 bus results.
    df = pd.read_csv('summary_stats_14.csv', index_col=0)

    for g, s in zip(ALL_GROUPS, GROUP_STRINGS):
        # Filter the DataFrame by entries in the group.
        mask = df['run'].isin(g)
        sub_df = df.loc[mask, :]
        assert sub_df.shape[0] == GROUP_LEN
        # Sort the DataFrame with a crappy loop.
        s_df = sub_df.copy(deep=True)
        s_df.reset_index(drop=True, inplace=True)
        idx = []
        for run in g:
            idx.append(np.argmax(s_df['run'] == run))
        s_df = s_df.loc[idx, :]
        s_df.reset_index(drop=True, inplace=True)

        # Okay, we have a sorted DataFrame for which we can start
        # making bar charts for.
        fig, ax = plt.subplots()
        # Add lines for graph-based and random agents.
        graph_line = ax.axhline(
            y=GRAPH_SUCCESS_OOB * 100, color='tab:orange', zorder=1,
            linewidth=2.0, linestyle='--')
        random_line = ax.axhline(
            y=RANDOM_SUCCESS_OOB * 100, color='tab:purple', zorder=1,
            linewidth=3.0, linestyle=':')
        # Get colors for bar chart.
        color_map = plt.get_cmap('tab10', 5)
        bars = ax.bar(x=[0, 1, 2, 3, 4],
                      height=(s_df['pct_success_oob'] * 100).to_numpy(),
                      tick_label=[1, 2, 3, 4, 5], zorder=2,
                      color=color_map.colors)
        # Set y limits for consistency.
        ax.set_ybound(0.0, 50.0)
        ax.set_xlabel(
            'Experiments with Different Observations, Rewards, and Algorithms')
        ax.set_ylabel('Mean O.O.B. Success Percentage Across Three Runs')
        ax.grid(True, which='major', axis='y')
        ax.set_title('Mean O.O.B. Success Rates for Different Experiments'
                     + f'\nObservations: {s}')
        add_value_labels(ax)
        ax.set_axisbelow(True)
        # ax.legend((graph_line, random_line), ('Graph Agent', 'Random Agent'))
        # Shrink current axis
        # https://stackoverflow.com/a/4701285/11052174
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.99, box.height])
        ax.legend((graph_line, random_line, *bars),
                  ('Graph\nAgent', 'Random\nAgent', 'P.U. Volt.,\nRew. 1',
                   'P.U. Volt.,\nMod. Alg.,\nRew. 1', 'M.M. Volt.,\nRew. 1',
                   'M.M. Volt., \nMod. Alg.\nRew. 1',
                   'M.M. Volt., \nMod. Alg., \nRew. 2'),
                  loc='center left',
                  bbox_to_anchor=(1, 0.5), ncol=1,)
        plt.tight_layout()
        # plt.show()
        fig.savefig(s.replace(' ', '_').replace(',', '') + '.png',
                    format='png')
        fig.savefig(s.replace(' ', '_').replace(',', '') + '.eps',
                    format='eps')


if __name__ == '__main__':
    main()
