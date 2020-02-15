"""Initialize environments, do pre-screening of power flow cases."""
# noinspection PyUnresolvedReferences,PyPackageRequirements
from constants import ENV_DICT, MIN_LOAD_FACTOR_DEFAULT,\
    MAX_LOAD_FACTOR_DEFAULT, LOAD_ON_PROBABILITY_DEFAULT,\
    LEAD_PF_PROBABILITY_DEFAULT, NUM_TIME_STEPS_DEFAULT,\
    NUM_SCENARIOS_DEFAULT, NUM_RUNS_DEFAULT, IEEE_14_PWB,\
    get_file_str, SHUNT_CLOSED_PROBABILITY_DEFAULT, \
    NUM_GEN_VOLTAGE_BINS_DEFAULT, GEN_VOLTAGE_RANGE_DEFAULT, \
    LOW_V_DEFAULT, HIGH_V_DEFAULT, TRUNCATE_VOLTAGES_DEFAULT, \
    MIN_LOAD_PF_DEFAULT
import gym
# Must import gym_powerworld for the environments to get registered.
# noinspection PyUnresolvedReferences
import gym_powerworld
from gym_powerworld.envs.voltage_control_env import OutOfScenariosError
import pickle
import multiprocessing as mp
from copy import deepcopy
import os
import logging
import threading
import json

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# Fill in the ENV_DICT with defaults.
ENV_DICT_FILLED = deepcopy(ENV_DICT)
ENV_DICT_FILLED['num_scenarios'] = NUM_SCENARIOS_DEFAULT
ENV_DICT_FILLED['max_load_factor'] = MAX_LOAD_FACTOR_DEFAULT
ENV_DICT_FILLED['min_load_factor'] = MIN_LOAD_FACTOR_DEFAULT
ENV_DICT_FILLED['min_load_pf'] = MIN_LOAD_PF_DEFAULT
ENV_DICT_FILLED['lead_pf_probability'] = LEAD_PF_PROBABILITY_DEFAULT
ENV_DICT_FILLED['load_on_probability'] = LOAD_ON_PROBABILITY_DEFAULT

# Keep the log level low.
ENV_DICT_FILLED['log_level'] = logging.ERROR

LOG.setLevel(logging.INFO)


def run(seed, v_truncate, case_path, env_id, case_str, log_queue):
    """
    :param seed: Seed for random numbers.
    :param v_truncate: Boolean passed to environment for forcing
        voltages within certain band.
    :param case_path: Full path to the PowerWorld case.
    :param env_id: String corresponding to the ID of the environment to
        use. Passed directly to gym.make.
    :param case_str: Identifying string for a case. E.g. '14' for
        14 bus.
    :param log_queue: Queue for logging. Put messages here.
    """

    # Make a copy of the ENV_DICT.
    ed = deepcopy(ENV_DICT_FILLED)

    # Update the seed, voltage truncation, and case path.
    ed['seed'] = seed
    ed['truncate_voltages'] = v_truncate
    ed['pwb_path'] = case_path

    # Use the following to encode files of various sorts.
    file_str = get_file_str(case_str=case_str, seed=seed,
                            v_truncate=v_truncate)

    # Train logfile will be a combination of the seed and v_truncate.
    log = 'log' + file_str + '.csv'
    ed['csv_logfile'] = log

    # Make the environment.
    env = gym.make(env_id, **ed)

    # Loop and run reset to set scenario and solve the power flow.
    log_thresh = ed['num_scenarios'] / 100
    log_incr = log_thresh
    while True:
        if env.scenario_idx >= log_thresh:
            pct = env.scenario_idx / ed['num_scenarios'] * 100
            log_thresh += log_incr
            log_queue.put(
                f'{file_str} is {pct:.1f}% complete ({env.scenario_idx} '
                'scenarios).')
        try:
            env.reset()
        except OutOfScenariosError:
            break

    # Pickle the 'scenario_init_success' parameter.
    with open('mask' + file_str + '.pkl', 'wb') as f:
        pickle.dump(env.scenario_init_success, f)

    # Dump the environment dict, making sure to pop the dtype.
    ed.pop('dtype')
    with open('env_input' + file_str + '.json', 'w') as f:
        json.dump(ed, f)

    # Remove the log.
    try:
        os.remove(log)
    except FileNotFoundError:
        pass

    # Close the environment.
    env.close()


def log_progress(q: mp.Queue):
    while True:
        m = q.get()
        if m is None:
            break
        LOG.info(m)


def main():
    # Create logging queue.
    lq = mp.Queue()

    # Create logging thread.
    lt = threading.Thread(target=log_progress, args=(lq,))
    lt.start()

    # Initialize list to hold our processes.
    processes = []

    # Loop and start processes.
    for seed in range(NUM_RUNS_DEFAULT):
        for v_truncate in [True, False]:
            for case_path, case_str in zip([IEEE_14_PWB], ['14']):
                for env_id in ['powerworld-discrete-env-simple-14-bus-v0']:
                    kwargs = {
                        'seed': seed,
                        'v_truncate': v_truncate,
                        'case_path': case_path,
                        'env_id': env_id,
                        'case_str': case_str,
                        'log_queue': lq
                    }
                    p = mp.Process(target=run, kwargs=kwargs)
                    processes.append(p)
                    p.start()

    # Wait.
    for p in processes:
        p.join()

    # Kill the logging thread.
    lq.put(None)


def test():
    with open('mask_14_0_0.pkl', 'rb') as f:
        arr = pickle.load(f)

    print(arr.shape)


if __name__ == '__main__':
    main()
    # test()
