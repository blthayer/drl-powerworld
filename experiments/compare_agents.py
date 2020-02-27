"""Perform the same tests, but with a random agent and a graph-based.
agent.
"""
# noinspection PyUnresolvedReferences,PyPackageRequirements
from constants import THIS_DIR, IEEE_14_PWB, IEEE_14_PWB_CONDENSERS, \
    IEEE_14_ONELINE_AXD, IEEE_14_CONTOUR_AXD, ENV_DICT, BASELINES_DICT, \
    MIN_LOAD_FACTOR_DEFAULT, MAX_LOAD_FACTOR_DEFAULT, \
    LOAD_ON_PROBABILITY_DEFAULT, LEAD_PF_PROBABILITY_DEFAULT, \
    NUM_TIME_STEPS_DEFAULT, NUM_SCENARIOS_DEFAULT, NUM_RUNS_DEFAULT, \
    get_file_str, MIN_LOAD_PF_DEFAULT, DATA_DIR, TEST_EPISODES, \
    IL_200_PWB, SC_500_PWB
import json
import numpy as np
import gym
# noinspection PyUnresolvedReferences
import gym_powerworld
import os
import pickle
# import shutil
import networkx as nx
import re
import argparse
from typing import Union

from gym_powerworld.envs.voltage_control_env import V_TOL


def main(case_str, random, mod, env_name, clipped_r, seed):
    # Primary output directory.
    if random:
        s = 'random'
    else:
        s = 'graph'

    if 'no-contingencies' in env_name:
        contingencies = False
        con_str = 'no_con'
    else:
        contingencies = True
        con_str = 'with_con'

    if not clipped_r:
        d = os.path.join(DATA_DIR, f'{s}_agent_{case_str}_{con_str}')
    else:
        d = os.path.join(DATA_DIR,
                         f'{s}_agent_clipped_reward_{case_str}_{con_str}')

    if mod:
        d = d + '_mod'

    # Get path to case. Needed for running on different machines with
    # different user names...
    if case_str == '14':
        case_path = IEEE_14_PWB
    elif case_str == '200':
        case_path = IL_200_PWB
    elif case_str == '500':
        case_path = SC_500_PWB
    else:
        raise NotImplementedError()

    try:
        os.mkdir(d)
    except FileExistsError:
        pass
        # shutil.rmtree(d)
        # os.mkdir(d)

    if seed is not None:
        iterable = [seed]
    else:
        iterable = range(NUM_RUNS_DEFAULT)

    # Loop.
    for seed in iterable:
        # Create directory.
        run_dir = os.path.join(d, f'run_{seed}')
        os.mkdir(run_dir)

        # Get a file string so we can load up the environment dict.

        fs = get_file_str(case_str=case_str, seed=seed,
                          contingencies=contingencies)
        with open(f'env_input{fs}.json', 'r') as f:
            env_dict = json.load(f)

        # Add the numpy data type.
        env_dict['dtype'] = np.float32

        # Adjust the log file.
        env_dict['csv_logfile'] = os.path.join(run_dir, 'log_test.csv')

        # Override case path.
        env_dict['pwb_path'] = case_path

        # Set the reward clipping.
        env_dict['clipped_reward'] = clipped_r

        # Load up the mask.
        with open(f'mask{fs}.pkl', 'rb') as f:
            mask = pickle.load(f)

        # Initialize the environment.
        env = gym.make(env_name, **env_dict)

        # Filter.
        env.filter_scenarios(mask)

        # Set scenario index for running tests, to be consistent with
        # the other agents.
        env.scenario_idx = env.num_scenarios - TEST_EPISODES - 1

        # Seed the action space if necessary.
        if random:
            env.action_space.seed(seed=seed)
        else:
            # Initialize a graph agent.
            g_agent = GraphAgent(env=env, file_str=fs, random=random)

        # Initialize array to track success.
        success = np.zeros(TEST_EPISODES)
        # Loop and take random actions.
        action_list = []

        for i in range(TEST_EPISODES):
            _ = env.reset()
            done = False

            if not random:
                # Prep the graph agent.
                # noinspection PyUnboundLocalVariable
                g_agent.init_episode()

            # Clear the action list.
            action_list.clear()

            while not done:
                if random:
                    # Get a random action for the agent.
                    action = get_random_action(env)

                    # If we're doing the 'mod' algorithm, sample until
                    # we get an action that is not in the list.
                    if mod:
                        while action in action_list:
                            action = get_random_action(env)

                else:
                    # Use the graph technique to get an action.
                    action = g_agent.get_action(action_list)

                # Put the action in the action list.
                action_list.append(action)

                # Perform the step.
                _, _, done, info = env.step(action)

                # TODO/NOTE: IMPORTANT! Breaking the loop early if the
                #   graph agent takes the no-op action. This will give
                #   the graph agent an artificially high reward, so
                #   it's important we NOT compare rewards and only
                #   compare success rates.
                if (not random) and (action == env.no_op_action):
                    # Since we're hacking around things, the environment
                    # hasn't actually filled in the 'is_success' field
                    # if we aren't done.
                    if not done:
                        info['is_success'] = False
                    done = True

            # noinspection PyUnboundLocalVariable
            success[i] = info['is_success']

        # Close environment, which flushes log.
        env.close()

        suc_pct = success.sum() / success.shape[0] * 100
        print(f'Success percentage: {suc_pct:.2f}%')


class GraphAgent:
    """Simple graph-based agent."""

    def __init__(self, env: gym_powerworld.envs.DiscreteVoltageControlEnv,
                 file_str, random):
        # Track environment.
        self.env = env

        # File string and random are used to make a unique Y-bus string.
        self.ybus_file = os.path.join(
            THIS_DIR, f'ybus_{file_str}_{int(random)}.mat')

        # Extract nodes with generators. This should come back
        # sorted since the gen_init_data comes back from ESA sorted
        # by BusNum.
        self.gen_buses = self.env.gen_buses.to_numpy()

        # Extract nodes with shunts.
        # TODO: IMPORTANT! This will NOT WORK when there are multiple
        #   shunts at a single bus. It'll work for now with the 200 and
        #   500 bus cases which do not have multiple shunts at the
        #   same buses. The fix is to do something similar inside the
        #   gym-powerworld repo as was done with generators.
        #   Multi-indexing and such.
        if self.env.num_shunts > 0:
            self.shunt_buses = self.env.shunt_init_data['BusNum'].to_numpy()
        else:
            self.shunt_buses = None

        # For the graph technique, we'll be setting gens to the
        # highest and second highest allowable voltage. Take
        # advantage of this fact, and compute the starting action
        # numbers for these voltage set points. The + 1 is due to
        # the no-op action.
        self.starting_action_num_105 = \
            self.env.num_gen_reg_buses * (self.env.gen_bins.shape[0] - 1) + 1

        self.starting_action_num_1025 = \
            self.env.num_gen_reg_buses * (self.env.gen_bins.shape[0] - 2) + 1

        # The following will be configured in the init_episode method.
        self.gens_on: Union[np.ndarray, None] = None
        self.shunts_on: Union[np.ndarray, None] = None
        self.graph: Union[nx.Graph, None] = None
        self.distance_indices = []
        self.prev_bus_num = None
        self.prev_distances: Union[np.ndarray, None] = None
        self.prev_distances_argsort = None

    def init_episode(self):
        """Perform initialization work required for an episode."""
        self.gens_on = self.gen_buses[self.env.gen_bus_status_arr]
        if self.shunt_buses is not None:
            self.shunts_on = self.shunt_buses[self.env.shunt_status_arr]
        self.graph = self.get_network_graph()
        self.distance_indices.clear()
        self.prev_bus_num = None
        self.prev_distances = None
        self.prev_distances_argsort = None

    def get_action(self, action_list):
        # Before proceeding any further, check to see if all voltages are
        # already in bounds. If this is the case, take the no-op action.
        if self.env.all_v_in_range:
            return 0

        # Pull the bus voltages from the environment.
        obs = self.env.bus_pu_volt_arr

        # Get lowest and highest magnitude voltage buses, recalling that
        # the bus voltages always go first in the observation. Bump by 1 due
        # to 0-based indexing.
        low_bus = int(np.argmin(obs) + 1)
        low_v = obs[low_bus - 1]
        high_bus = int(np.argmax(obs) + 1)
        high_v = obs[high_bus - 1]

        # Set flag for whether or not we're going to work on the lowest
        # voltage or the highest voltage.
        low_flag = low_v < (0.95 + V_TOL)

        # Determine which bus to get electrical distances from, and which
        # action number to use in computing the action to take.
        if low_flag:
            bus = low_bus
            starting_action_num = self.starting_action_num_105
            v = 1.05
        elif high_v > (1.05 + V_TOL):
            bus = high_bus
            starting_action_num = self.starting_action_num_1025
            v = 1.025
        else:
            # We should never get here.
            return 0

        # Get electrical distances from the bus to each generator (reactance
        # only). Skip if we're looking at the same bus.
        if bus == self.prev_bus_num:
            distances = self.prev_distances
            distances_argsort = self.prev_distances_argsort
        else:
            distances = np.zeros(self.gens_on.shape[0])
            for idx in range(self.gens_on.shape[0]):
                try:
                    d = nx.dijkstra_path_length(
                        self.graph, bus, self.gens_on[idx], 'x')
                except nx.exception.NetworkXNoPath:
                    # If for whatever reason there's no path, set the
                    # distance to infinity.
                    d = np.inf
                    print('No path!')

                distances[idx] = d

            # Track bus and distances to possibly save computation in the
            # next step.
            self.prev_bus_num = bus
            self.prev_distances = distances
            distances_argsort = np.argsort(distances)
            self.prev_distances_argsort = distances_argsort

        # Loop to determine a valid action. Initialize
        # to the no-op action.
        action = 0
        for idx in distances_argsort:
            # Skip this generator if the distance is infinite.
            if np.isinf(distances[idx]):
                continue

            # Get the bus number corresponding to this
            # generator.
            gen_bus = self.gens_on[idx]

            # noinspection PyUnresolvedReferences
            gen_mask = \
                (self.env.gen_obs_data['BusNum'] == gen_bus).to_numpy()

            # Look up the voltage set point.
            v_set = \
                self.env.gen_obs_data['GenVoltSet'].to_numpy()[
                    gen_mask]
            # There can be multiple generators at a bus. The environments
            # should command them all to the same set point. Sanity check
            # that here.
            assert np.allclose(v_set[0], v_set)
            v_set = v_set[0]

            # If the set point is already at the appropriate setting, or
            # move on to the next generator.
            if np.isclose(v_set, v, rtol=0.0, atol=V_TOL):
                continue

            # Put this generator index in the list so we can skip it
            # next time.
            self.distance_indices.append(idx)

            # Determine what action we need.
            action = starting_action_num + np.argmax(gen_mask)

            # If this action has already been taken this episode, keep on
            # looping.
            if action in action_list:
                continue
            else:
                # We have an action to take. Break the loop.
                break

        return action

    def get_network_graph(self):
        # Initialize a graph.
        g = nx.Graph()

        # Save YBus.
        self.env.saw.RunScriptCommand(
            f'SaveYbusInMatlabFormat("{self.ybus_file}", NO)')

        # Load YBus.
        with open(self.ybus_file, 'r') as f1:
            # Pass on the first two lines. They contain something like:
            # j = sqrt(-1);
            # Ybus = sparse(14);
            f1.readline()
            f1.readline()
            # Read the rest.
            mat_str = f1.read()

        # Remove whitespace from the string.
        mat_str = re.sub('\s', '', mat_str)

        # Split by semicolons.
        entries = re.split(';', mat_str)

        # Loop. The last entry is the empty string, so exclude it.
        for e in entries[:-1]:
            # Split about the equals sign.
            eq_split = re.split('=', e)

            # The first part is going to be related to the index, the second
            # part related to the value.
            idx_str = eq_split[0]
            value_str = eq_split[1]

            # Pull the values out of the string. This is crude, but works.
            idx = [int(x) for x in
                   re.split(',',
                            re.match('(?:Ybus\()(.*)(?:\))', idx_str).group(
                                1))]

            # Not working with diagonals.
            if idx[0] == idx[1]:
                continue

            # Remove parentheses from the value_str.
            value_str = re.sub('(?:\()(.*)(?:\))', lambda m: m.group(1),
                               value_str)

            # Split by the '+j*'
            val_list = [float(x) for x in re.split('\+j\*', value_str)]

            # Create complex number, invert and multiply by -1 to convert
            # admittance to impedance.
            try:
                cplx = -1 / (val_list[0] + 1j * val_list[1])
            except ZeroDivisionError:
                # Move along if the buses are not connected.
                continue

            # Extract the reactance.
            x = cplx.imag

            # Add to the graph.
            g.add_edge(idx[0], idx[1], x=x)

        return g


def get_random_action(env):
    return env.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=str, choices=['14', '200', '500'],
                        help='Case to use.')
    parser.add_argument(
        'env', type=str,
        choices=['powerworld-discrete-env-simple-14-bus-v0',
                 'powerworld-discrete-env-gen-shunt-no-contingencies-v0',
                 'powerworld-discrete-env-gen-branch-shunt-v0',
                 ],
        help='Environment to use')
    parser.add_argument(
        '--random', action='store_true',
        help='Include this flag to use the random agent. Otherwise, the graph '
             'agent will be used.')
    parser.add_argument(
        '--mod', action='store_true',
        help='Include this flag to force the random agent to not take the same'
             ' action multiple times in an episode.'
    )
    parser.add_argument(
        '--clipped', action='store_true',
        help='Include this flag to use the clipped rewards.'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='If a seed is provided, a single run with the given seed is '
             'performed. If a seed is not provided, all seeds in range('
             f'{NUM_RUNS_DEFAULT}) will be run in series.')
    args_in = parser.parse_args()
    main(case_str=args_in.case, env_name=args_in.env, random=args_in.random,
         mod=args_in.mod, clipped_r=args_in.clipped, seed=args_in.seed)
    # Random:
    # main(case_str='14', random=True, mod=False,
    #      env_name='powerworld-discrete-env-simple-14-bus-v0',
    #      clipped_r=False)
    # main(case_str='14', random=True, mod=True,
    #      env_name='powerworld-discrete-env-simple-14-bus-v0',
    #      clipped_r=False)
    # main(case_str='14', random=True, mod=False,
    #      env_name='powerworld-discrete-env-simple-14-bus-v0',
    #      clipped_r=True)
    # main(case_str='14', random=True, mod=True,
    #      env_name='powerworld-discrete-env-simple-14-bus-v0',
    #      clipped_r=True)
    # Graph:
    # main(case_str='14', random=False, mod=False,
    #      env_name='powerworld-discrete-env-simple-14-bus-v0',
    #      clipped_r=False)
    # main(case_str='14', random=False, mod=False,
    #      env_name='powerworld-discrete-env-simple-14-bus-v0',
    #      clipped_r=True)
