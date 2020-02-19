"""Perform the same tests, but with a random agent and a graph-based.
agent.

TODO/NOTE: At present, the graph-based approach will only work for the
    14 bus case.
"""
# noinspection PyUnresolvedReferences,PyPackageRequirements
from constants import THIS_DIR, IEEE_14_PWB, IEEE_14_PWB_CONDENSERS, \
    IEEE_14_ONELINE_AXD, IEEE_14_CONTOUR_AXD, ENV_DICT, BASELINES_DICT, \
    MIN_LOAD_FACTOR_DEFAULT, MAX_LOAD_FACTOR_DEFAULT, \
    LOAD_ON_PROBABILITY_DEFAULT, LEAD_PF_PROBABILITY_DEFAULT, \
    NUM_TIME_STEPS_DEFAULT, NUM_SCENARIOS_DEFAULT, NUM_RUNS_DEFAULT, \
    get_file_str, MIN_LOAD_PF_DEFAULT, DATA_DIR, TEST_EPISODES
import json
import numpy as np
import gym
# noinspection PyUnresolvedReferences
import gym_powerworld
import os
import pickle
import shutil
import networkx as nx
import re

from gym_powerworld.envs.voltage_control_env import V_TOL


def main(case_str, random, mod, env_name):
    # Primary output directory.
    if random:
        s = 'random'
    else:
        s = 'graph'

    d = os.path.join(DATA_DIR, f'{s}_agent_{case_str}')

    if mod:
        d = d + '_mod'

    # Get path to case. Needed for running on different machines with
    # different user names...
    if case_str == '14':
        case_path = IEEE_14_PWB
    else:
        raise NotImplementedError()

    try:
        os.mkdir(d)
    except FileExistsError:
        shutil.rmtree(d)
        os.mkdir(d)

    # Loop.
    for seed in range(NUM_RUNS_DEFAULT):
        # Create directory.
        run_dir = os.path.join(d, f'run_{seed}')
        os.mkdir(run_dir)

        # Get a file string so we can load up the environment dict.
        fs = get_file_str(case_str=case_str, seed=seed, v_truncate=True)
        with open(f'env_input{fs}.json', 'r') as f:
            env_dict = json.load(f)

        # Add the numpy data type.
        env_dict['dtype'] = np.float32

        # Adjust the log file.
        env_dict['csv_logfile'] = os.path.join(run_dir, 'log_test.csv')

        # Override case path.
        env_dict['pwb_path'] = case_path

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
            # Get a graph of the network.
            graph = get_network_graph(env)

            # Extract nodes with generators. Ensure list is sorted.
            gen_buses = [x for x, y in graph.nodes(data=True) if y['gen']]
            gen_buses.sort()

            # For the graph technique, we'll be setting gens to the
            # highest allowable voltage. Take advantage of this fact,
            # and compute the lowest action number. The + 1 is due to
            # the no-op action.
            starting_action_num = \
                env.num_gens * (env.gen_bins.shape[0] - 1) + 1

        # Loop and take random actions.
        action_list = []
        for _ in range(TEST_EPISODES):
            obs = env.reset()
            done = False

            if not random:
                # Get generators which are on.
                # noinspection PyUnboundLocalVariable
                gens_on = np.array(gen_buses)[env.gen_status_arr.astype(bool)]

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
                    # noinspection PyUnboundLocalVariable
                    action = get_graph_action(
                        env=env, graph=graph, obs=obs, gens_on=gens_on,
                        starting_action_num=starting_action_num)

                # Put the action in the action list.
                action_list.append(action)

                # Perform the step.
                obs, _, done, _ = env.step(action)

        # Close environment, which flushes log.
        env.close()


def get_random_action(env):
    return env.action_space.sample()


def get_graph_action(env, graph, obs, gens_on, starting_action_num):
    # TODO: Update to work with cases other than the 14 node.

    # Get lowest magnitude voltage bus, recalling that
    # the bus voltages always go first in the
    # observation. Bump by 1 due to 0-based indexing.
    low_bus = int(np.argmin(obs[0:env.num_buses]) + 1)
    low_v = obs[0:env.num_buses][low_bus - 1]

    # If the lowest voltage is above the threshold,
    # take the no-op action.
    if low_v > (0.95 + V_TOL):
        action = 0
    else:
        # Get electrical distances from the lowest bus
        # to each generator (reactance only).
        distances = \
            np.array(
                [nx.dijkstra_path_length(
                    graph, low_bus, gen_bus, 'x')
                    for gen_bus in gens_on])

        # Loop to determine a valid action. Initialize
        # to the no-op action.
        action = 0
        for idx in np.argsort(distances):
            # Get the bus number corresponding to this
            # generator.
            gen_bus = gens_on[idx]

            # noinspection PyUnresolvedReferences
            gen_mask = \
                (env.gen_obs_data['BusNum'] == gen_bus).to_numpy()

            # Look up the voltage set point.
            # TODO: The next few lines will likely cause
            #   issues with cases other than the 14 bus.
            v_set = \
                env.gen_obs_data['GenVoltSet'].to_numpy()[
                    gen_mask]
            assert v_set.shape[0] == 1
            v_set = v_set[0]

            # If the set point is already at the max,
            # move on to the next generator.
            if np.isclose(v_set, 1.05, rtol=0.0, atol=V_TOL):
                continue

            # Determine what action we need and stop looping.
            action = starting_action_num + np.argmax(gen_mask)
            break

    return action


def get_network_graph(env):
    """Get a graph representing the PowerWorld case. The resulting graph
    will have edges with an 'x' attribute for reactance (float), and
    nodes will have a 'gen' attribute (True or False)
    """
    # Initialize a graph.
    g = nx.Graph()

    # Save YBus.
    f = os.path.join(THIS_DIR, 'ybus.mat')
    env.saw.RunScriptCommand(f'SaveYbusInMatlabFormat("{f}", NO)')
    # Load YBus.
    with open(f, 'r') as f1:
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
               re.split('Ybus\(', idx_str)[1].replace(')', '').split(',')]

        # Not working with diagonals.
        if idx[0] == idx[1]:
            continue

        # Remove parentheses from the value_str.
        value_str = value_str.replace(')', '').replace('(', '')

        # Split by the '+j*'
        val_list = [float(x) for x in re.split('\+j\*', value_str)]

        # Create complex number, invert and multiply by -1 to convert
        # admittance to impedance.
        cplx = -1 / (val_list[0] + 1j * val_list[1])

        # Extract the reactance.
        x = cplx.imag

        # Add to the graph.
        g.add_edge(idx[0], idx[1], x=x)

    # Extract generator data.
    kf = env.saw.get_key_field_list('gen')
    gens = env.saw.GetParametersMultipleElement(
        ObjectType='gen', ParamList=kf
    )
    # Get a set of generator buses.
    gen_buses = set(gens['BusNum'].to_numpy(dtype=int))

    # Add generators to nodes.
    for n in g.nodes:
        g.nodes[n]['gen'] = n in gen_buses

    return g


if __name__ == '__main__':
    # main(case_str='14', random=True, mod=False,
    #      env_name='powerworld-discrete-env-simple-14-bus-v0')
    # main(case_str='14', random=True, mod=True,
    #      env_name='powerworld-discrete-env-simple-14-bus-v0')
    main(case_str='14', random=False, mod=False,
         env_name='powerworld-discrete-env-simple-14-bus-v0')
