"""Perform the same tests, but with a random agent and a graph-based.
agent.
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


def main(case_str, random, mod, env_name):
    # Primary output directory.
    d = os.path.join(DATA_DIR, f'random_agent_{case_str}')

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

        # Loop and take random actions.
        action_list = []
        for _ in range(TEST_EPISODES):
            env.reset()
            done = False

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
                    raise NotImplementedError()

                # Put the action in the action list.
                action_list.append(action)

                # Perform the step.
                _, _, done, _ = env.step(action)

        # Close environment, which flushes log.
        env.close()


def get_random_action(env):
    return env.action_space.sample()


if __name__ == '__main__':
    main(case_str='14', random=True, mod=False,
         env_name='powerworld-discrete-env-simple-14-bus-v0')
    main(case_str='14', random=True, mod=True,
         env_name='powerworld-discrete-env-simple-14-bus-v0')
