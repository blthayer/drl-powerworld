import os
import tensorflow as tf
import gym
# Must import gym_powerworld for the environments to get registered.
# noinspection PyUnresolvedReferences
import gym_powerworld
import numpy as np
import time
import shutil
from copy import deepcopy
import json
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN
import argparse
# noinspection PyUnresolvedReferences
from dqn_mod import DQNUniqueActions, build_act_mod, step_mod
from unittest.mock import patch
import pickle

# noinspection PyUnresolvedReferences,PyPackageRequirements
from constants import THIS_DIR, IEEE_14_PWB, IEEE_14_PWB_CONDENSERS, \
    IEEE_14_ONELINE_AXD, IEEE_14_CONTOUR_AXD, ENV_DICT, BASELINES_DICT, \
    MIN_LOAD_FACTOR_DEFAULT, MAX_LOAD_FACTOR_DEFAULT, \
    LOAD_ON_PROBABILITY_DEFAULT, LEAD_PF_PROBABILITY_DEFAULT, \
    NUM_TIME_STEPS_DEFAULT, NUM_SCENARIOS_DEFAULT, NUM_RUNS_DEFAULT, \
    get_file_str, MIN_LOAD_PF_DEFAULT, DATA_DIR, TEST_EPISODES


def callback_factory(average_reward, max_episodes):

    def callback(lcl, _glb) -> bool:
        """
        :param lcl: locals() inside deepq.learn
        :param _glb: globals() inside deepq.learn
        """
        # Compute the average of the last 100 episodes.
        avg_100 = sum(lcl['episode_rewards'][-101:-1]) / 100

        # Length indicates how many episodes/scenarios we've gone
        # through.
        num_ep = len(lcl['episode_rewards'])

        if avg_100 >= average_reward:
            # Terminate training.
            print('Terminating training since either the 100 episode average '
                  f'reward has exceeded {average_reward}.')
            return False
        elif num_ep >= max_episodes:
            # Terminate.
            print(f'Terminating training since we have hit {num_ep} episodes.')
            return False
        else:
            # Don't terminate training.
            return True

    return callback


def learn_and_test(out_dir, seed, env_name, num_scenarios, num_time_steps,
                   callback, policy, case, max_load_factor,
                   min_load_factor, lead_pf_probability,
                   load_on_probability, mod_learn, v_truncate, case_str,
                   scale_v_obs, clipped_r, gamma):
    """Use this function to take a shot at replicating the GridMind
    paper: https://arxiv.org/abs/1904.10597

    Use the "condensers" case because it closely represents the case
    they used.
    """

    # TODO: Put images on the SSD so it runs faster.
    # Files and such.
    image_dir = os.path.join(out_dir, 'images')
    train_logfile = os.path.join(out_dir, 'log_train.csv')
    test_logfile = os.path.join(out_dir, 'log_test.csv')
    model_file = os.path.join(out_dir, 'gridmind_reproduce.pkl')
    info_file = os.path.join(out_dir, 'info.txt')

    # Get a copy of the default inputs.
    env_dict = deepcopy(ENV_DICT)

    # overwrite the seed, image_dir, and csv_logfile.
    env_dict['pwb_path'] = case
    env_dict['seed'] = seed
    env_dict['image_dir'] = image_dir
    env_dict['csv_logfile'] = train_logfile
    # Set remaining inputs.
    env_dict['num_scenarios'] = num_scenarios
    env_dict['max_load_factor'] = max_load_factor
    env_dict['min_load_factor'] = min_load_factor
    env_dict['lead_pf_probability'] = lead_pf_probability
    env_dict['load_on_probability'] = load_on_probability
    env_dict['truncate_voltages'] = v_truncate
    env_dict['scale_voltage_obs'] = scale_v_obs
    env_dict['clipped_reward'] = clipped_r

    # Initialize the environment.
    env = gym.make(env_name, **env_dict)

    # See if we've pre-screened for this input combination.
    # TODO: Update for contingencies. Need an input or some other way of discriminating.
    file_str = get_file_str(case_str=case_str, seed=seed,
                            contingencies=True)

    json_file = 'env_input' + file_str + '.json'
    mask_file = 'mask' + file_str + '.pkl'
    try:
        with open(json_file, 'r') as f:
            env_dict_screen = json.load(f)
    except FileNotFoundError:
        env_dict_screen = None

    print('*' * 120)
    print('*' * 120)
    all_match = False
    screen = (('gridmind' not in env_name)
              or (env_name == 'powerworld-gridmind-hard-env-v0'))
    if screen:
        all_match = True
        # Compare the important params.
        for key in ['pwb_path', 'num_scenarios', 'max_load_factor',
                    'min_load_factor', 'min_load_pf', 'lead_pf_probability',
                    'load_on_probability', 'shunt_closed_probability',
                    'num_gen_voltage_bins', 'gen_voltage_range', 'seed',
                    'low_v', 'high_v', 'truncate_voltages']:
            # Extract params.
            p1 = env_dict_screen[key]
            p2 = env_dict[key]

            # If either p1 or p2 are lists, cast them to tuples.
            if isinstance(p1, list):
                p1 = tuple(p1)

            if isinstance(p2, list):
                p2 = tuple(p2)

            if p1 != p2:
                print(f'Cannot use screening vector because {key} params do '
                      f'not match. Screen: {p1}, Train: {p2}')
                all_match = False
                break

        if all_match:
            print('Key parameters matched up, so we can use the screening '
                  'vector.')
            with open(mask_file, 'rb') as f:
                screen_vec = pickle.load(f)

            env.filter_scenarios(screen_vec)
            k = screen_vec.sum()
            r = (~screen_vec).sum()
            print(f'Screening successful. Kept {k} scenarios, removed {r}.')
    else:
        print(f'Could not find file {json_file} so we cannot use the '
              'screening vector.')

    print('*' * 120)
    print('*' * 120)

    # Get a copy of the default inputs for dqn.
    init_dict = deepcopy(BASELINES_DICT)

    # Log inputs.
    env_dict.pop('dtype')
    with open(os.path.join(out_dir, 'env_input.json'), 'w') as f:
        json.dump(env_dict, f)

    # Overwrite seed and env
    init_dict['seed'] = seed
    init_dict['env'] = env
    init_dict['policy'] = policy
    init_dict['gamma'] = gamma

    # Initialize.
    if mod_learn:
        with patch('stable_baselines.deepq.build_graph.build_act',
                   new=build_act_mod):
            model = DQNUniqueActions(**init_dict)
    else:
        model = DQN(**init_dict)

    # Log inputs.
    init_dict.pop('policy')
    init_dict.pop('env')
    with open(os.path.join(out_dir, 'dqn_input.json'), 'w') as f:
        json.dump(init_dict, f)

    # Learning time.
    t0 = time.time()
    learn_dict = {'total_timesteps': num_time_steps, 'callback': callback,
                  'log_interval': 100}
    model.learn(**learn_dict)
    t1 = time.time()

    print('All done, saving to file.')
    model.save(model_file)

    # Save info file.
    s = f'Training took {t1-t0:.2f} seconds.\n'
    print(s)
    with open(info_file, 'w') as f:
        f.write(s)

    # Run through several "test" scenarios without training. Uncomment
    # the "render" lines to save images and display them.
    # Start by resetting the log (which will also flush the log).
    env.reset_log(new_file=test_logfile)

    # If we're using pre-screened scenarios, we know that we can count
    # on all power flows to be successful. So, we can confidently set
    # the scenario index to TEST_EPISODES episodes before the end.
    if all_match and screen:
        env.scenario_idx = env.num_scenarios - TEST_EPISODES - 1

    if mod_learn:
        test_loop_mod(env, model)
    else:
        test_loop(env, model)

    # Close the environment (which will flush the log).
    env.close()


def test_loop(env, model):
    for _ in range(TEST_EPISODES):
        obs = env.reset()
        done = False

        while not done:
            # env.render()
            obs, rew, done, _ = \
                env.step(model.predict(obs, deterministic=True)[0])


def test_loop_mod(env, model):
    action_list = list()
    for _ in range(TEST_EPISODES):
        # Get the environment ready.
        obs = env.reset()
        done = False

        # Clear the action list.
        action_list.clear()

        while not done:
            # env.render()
            # Get the ranked actions.
            action_arr = model.predict(obs, deterministic=True)[0]

            # Get the best action that has not yet been used this
            # episode, add it to the list.
            action = action_arr[np.argmin(np.isin(action_arr, action_list))]
            action_list.append(action)

            # Take the step.
            obs, rew, done, _ = env.step(action)


def loop(out_dir, env_name, runs, hidden_list, num_scenarios,
         avg_reward, num_time_steps, case, min_load_factor,
         max_load_factor, lead_pf_probability, load_on_probability,
         mod_learn, v_truncate, case_str, scale_v_obs, clipped_r, gamma):
    """Run the gridmind_reproduce function in a loop."""
    base_dir = os.path.join(DATA_DIR, out_dir)

    # Make the directory. Don't worry if it exists already.
    try:
        os.mkdir(base_dir)
    except FileExistsError:
        pass

    # Create the callback.
    callback = callback_factory(average_reward=avg_reward,
                                max_episodes=num_scenarios)

    # Create a custom policy using the specified layers.
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            # noinspection PyUnresolvedReferences
            super(CustomPolicy, self).__init__(
                *args, **kwargs, layers=hidden_list, layer_norm=False,
                feature_extraction='mlp', act_fun=tf.nn.relu)

    # If we're using the modified learning where each episode is only
    # allowed to take each action at most once, override the step
    # method.
    if mod_learn:
        CustomPolicy.step = step_mod

    # Loop over the runs.
    for i in range(runs):
        seed = i
        # Create name of directory for this run.
        tmp_dir = os.path.join(base_dir, f'run_{i}')

        # Attempt to create the directory, otherwise, delete.
        try:
            os.mkdir(tmp_dir)
        except FileExistsError:
            # Delete the directory if it's present, and remake it.
            shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)

        # Do the learning and testing.
        learn_and_test(
            out_dir=tmp_dir, seed=seed, env_name=env_name,
            num_scenarios=num_scenarios, num_time_steps=num_time_steps,
            callback=callback, policy=CustomPolicy, case=case,
            min_load_factor=min_load_factor,
            lead_pf_probability=lead_pf_probability,
            load_on_probability=load_on_probability,
            max_load_factor=max_load_factor, mod_learn=mod_learn,
            v_truncate=v_truncate, case_str=case_str, scale_v_obs=scale_v_obs,
            clipped_r=clipped_r, gamma=gamma
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', help='Relative output directory.',
                        type=str)
    parser.add_argument(
        'env', help='Gym PowerWorld environment to use.', type=str,
        choices=[
            'powerworld-gridmind-env-v0',
            'powerworld-gridmind-contingencies-env-v0',
            'powerworld-gridmind-hard-env-v0',
            'powerworld-discrete-env-simple-14-bus-v0',
            'powerworld-discrete-env-gen-state-14-bus-v0',
            'powerworld-discrete-env-branch-state-14-bus-v0',
            'powerworld-discrete-env-branch-and-gen-state-14-bus-v0'
        ])
    
    parser.add_argument(
        'case', help='Case to use.', type=str, choices=['14', '14_condensers'])
    parser.add_argument('--num_runs', help='Number of times to train.',
                        type=int, default=NUM_RUNS_DEFAULT)
    # https://stackoverflow.com/a/24866869/11052174
    parser.add_argument('--hidden_list',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default=[64, 64],
                        help='List of hidden layer sizes, e.g. "64,128"')
    parser.add_argument(
        '--num_scenarios', type=int, default=int(NUM_SCENARIOS_DEFAULT),
        help='Number of scenarios for the environment to create.',
    )
    parser.add_argument(
        '--avg_reward', type=float, default=198.75,
        help='Stop training when the 100 episode average has hit this reward.'
    )
    parser.add_argument(
        '--num_time_steps', type=int, default=int(NUM_TIME_STEPS_DEFAULT),
        help=('Number of time steps to run training (unless terminated early '
              'by achieving avg_reward). Note that the exploration rate is '
              'currently set to decay linearly from start to num_time_steps.'))
    parser.add_argument(
        '--mod_learn', action='store_true'
    )

    parser.add_argument('--min_load_factor', type=float,
                        default=MIN_LOAD_FACTOR_DEFAULT)
    parser.add_argument('--max_load_factor', type=float,
                        default=MAX_LOAD_FACTOR_DEFAULT)
    parser.add_argument('--load_on_probability', type=float,
                        default=LOAD_ON_PROBABILITY_DEFAULT)
    parser.add_argument('--lead_pf_probability', type=float,
                        default=LEAD_PF_PROBABILITY_DEFAULT)
    parser.add_argument('--min_load_pf', type=float,
                        default=MIN_LOAD_PF_DEFAULT)
    parser.add_argument('--v_truncate', action='store_true')
    parser.add_argument('--scale_v_obs', action='store_true')
    parser.add_argument('--clipped_r', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99)

    # Parse the arguments.
    args_in = parser.parse_args()

    if args_in.case == '14':
        case_ = IEEE_14_PWB
        case_str_ = args_in.case
    elif args_in.case == '14_condensers':
        case_ = IEEE_14_PWB_CONDENSERS
        case_str_ = args_in.case
    else:
        raise UserWarning('What is going on?')

    # Run.
    loop(out_dir=args_in.out_dir, env_name=args_in.env, runs=args_in.num_runs,
         hidden_list=args_in.hidden_list, num_scenarios=args_in.num_scenarios,
         avg_reward=args_in.avg_reward, num_time_steps=args_in.num_time_steps,
         case=case_, min_load_factor=args_in.min_load_factor,
         max_load_factor=args_in.max_load_factor,
         load_on_probability=args_in.load_on_probability,
         lead_pf_probability=args_in.lead_pf_probability,
         mod_learn=args_in.mod_learn, v_truncate=args_in.v_truncate,
         case_str=case_str_, scale_v_obs=args_in.scale_v_obs,
         clipped_r=args_in.clipped_r, gamma=args_in.gamma)
