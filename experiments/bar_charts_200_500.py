"""Read individual summary files to create grouped bar charts."""
import numpy as np
import pandas as pd
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
from matplotlib import cm
# noinspection PyUnresolvedReferences,PyPackageRequirements
from constants import add_value_labels, DATA_DIR
mpl.rcParams['savefig.dpi'] = 300

# Results directories
DIR_200_NO_CON = {
    'Random Agent': 'random_agent_200_no_con_mod',
    'Graph-Based Agent': 'graph_agent_200_no_con',
    'DRL Agent': 'de_200_branch_and_gen_state_v_scaled_mod_1024_1024'
}

DIR_200_WITH_CON = {
    'Random Agent': 'random_agent_200_with_con_mod',
    'Graph-Based Agent': 'graph_agent_200_with_con',
    'DRL Agent': 'de_200_con_branch_and_gen_state_v_scaled_mod_1024_1024'
}

DIR_500_NO_CON = {
    'Random Agent': 'random_agent_500_no_con_mod',
    'Graph-Based Agent': 'graph_agent_500_no_con',
    'DRL Agent': 'de_500_branch_and_gen_state_v_scaled_mod_2048_2048'
}

DIR_500_WITH_CON = {
    'Random Agent': 'random_agent_500_with_con_mod',
    'DRL Agent': 'de_500_con_branch_and_gen_state_v_scaled_mod_2048_2048',
    'Graph-Based Agent': 'graph_agent_500_with_con'
}


def main():
    # Loop over dictionaries.
    dirs = [DIR_200_NO_CON, DIR_200_WITH_CON, DIR_500_NO_CON,
            DIR_500_WITH_CON]
    titles = ['200 Bus System, No Contingencies',
              '200 Bus System, With Contingencies',
              '500 Bus System, No Contingencies',
              '500 Bus System, With Contingencies',
              ]
    for d, t in zip(dirs, titles):
        # Loop over directories.
        df = pd.DataFrame(data=np.zeros((3, 3)), columns=list(d.keys()))
        for key, directory in d.items():
            p = os.path.join(DATA_DIR, directory, 'summary.csv')
            data = pd.read_csv(p, index_col=0)
            try:
                df[key] = data['Success Percentage Start OOB']
            except KeyError:
                df[key] = data['Success % O.O.B.']

        make_bar(df, t)


def make_bar(df, t):
    fig, ax = plt.subplots()
    ind = np.array([0, 1, 2])
    width = 0.2
    rand_bar = ax.bar(
        x=(ind - width),
        height=((df['Random Agent'] * 100).to_numpy()), width=width,
        tick_label=ind, color='tab:purple', align='edge', label='Random Agent')

    drl_bar = ax.bar(
        x=ind,
        height=((df['DRL Agent'] * 100).to_numpy()), width=width,
        tick_label=ind, color='tab:grey', align='edge', label='DRL Agent')

    graph_bar = ax.bar(
        x=(ind + width),
        height=((df['Graph-Based Agent'] * 100).to_numpy()), width=width,
        tick_label=ind, color='tab:orange', align='edge', label='Graph Agent')

    # Set y limits for consistency.
    if '200' in t:
        yb = 45
    elif '500' in t:
        yb = 65
    else:
        raise UserWarning('hmmm')

    ax.set_ybound(0.0, yb)
    add_value_labels(ax)
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('O.O.B. Success Percentage')
    ax.set_title('O.O.B. Success Percentages for Different Agents'
                 + f'\n{t}')
    ax.grid(True, which='major', axis='y')
    ax.set_axisbelow(True)
    ax.legend(ncol=3)

    fig.savefig(t.replace(' ', '_').replace(',', '_') + '.png', format='png')
    fig.savefig(t.replace(' ', '_').replace(',', '_') + '.eps', format='eps')


if __name__ == '__main__':
    main()
