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

    # "line" graph for action_taken.
    # Relatively useless
    # plt.plot(df['action_taken'])

    # "bar" graph of all actions
    action_count = df_test['action_taken'].value_counts(
        normalize=False, sort=False, ascending=False, dropna=True)
    action_count_norm = df_test['action_taken'].value_counts(
        normalize=True, sort=True, ascending=False, dropna=True
    )
    # Still have to sort it? Fine.
    action_count.sort_index(inplace=True)
    # Bar graph.
    #
    plt.bar(x=action_count.index, height=action_count.to_numpy())
    plt.show()
    pass


if __name__ == '__main__':
    main()
