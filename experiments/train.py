import os
import tensorflow as tf
import gym
# Must import gym_powerworld for the environments to get registered.
# noinspection PyUnresolvedReferences
import gym_powerworld
import logging
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

# noinspection PyUnresolvedReferences,PyPackageRequirements
from constants import THIS_DIR, IEEE_14_PWB, IEEE_14_PWB_CONDENSERS, \
    IEEE_14_ONELINE_AXD, IEEE_14_CONTOUR_AXD

# Dictionary of GridMind environment inputs.
ENV_DICT = dict(
    # Five voltage bins: [0.95, 0.975, 1.0, 1.025, 1.05]
    num_gen_voltage_bins=5,
    gen_voltage_range=(0.95, 1.05),
    log_level=logging.INFO,
    # Use the same reward values.
    # rewards=dict(normal=100, violation=-50, diverged=-100),
    # Use Numpy float32.
    dtype=np.float32,
    # 0.95-1.05 is the "good" voltage range.
    low_v=0.95, high_v=1.05,
    # For later visualization:
    oneline_axd=IEEE_14_ONELINE_AXD,
    contour_axd=IEEE_14_CONTOUR_AXD,
    # Use a really small render interval so the "testing"
    # scenarios will go by quickly.
    render_interval=1e-9,
    # .csv logging:
    log_buffer=10000,
    # The following fields should be added in the function:
    # pwb_path=IEEE_14_PWB_CONDENSERS,
    # seed=seed,
    # num_scenarios=num_scenarios
    # max_load_factor=1.2
    # min_load_factor=0.8
    # lead_pf_probability=None
    # load_on_probability=None
    # image_dir=image_dir,
    # csv_logfile=train_logfile,
)

BASELINES_DICT = dict(
    # The following are all defaults:
    learning_rate=5e-4,
    buffer_size=50000,
    exploration_initial_eps=1.0,
    train_freq=1,
    batch_size=32,
    double_q=True,
    learning_starts=1000,
    target_network_update_freq=500,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta0=0.4,
    prioritized_replay_beta_iters=None,
    prioritized_replay_eps=1e-6,
    param_noise=False,
    tensorboard_log=None,
    _init_setup_model=True,
    full_tensorboard_log=False,
    n_cpu_tf_sess=None,
    # Not default:
    verbose=1,
    prioritized_replay=True,
    # Go all the way down to 1%.
    exploration_final_eps=0.01,
    # Set gamma to 1.0, since there are already incentives built-in for
    # minimizing the number of actions.
    gamma=1.0,
    # Have exploration linearly decay based on total_timesteps.
    exploration_fraction=1.0,
    # Update the following:
    # policy=policy
    # seed=seed
    # env=env
)


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
                   load_on_probability, mod_learn):
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
    env_dict['num_scenarios'] = num_scenarios
    env_dict['max_load_factor'] = max_load_factor
    env_dict['min_load_factor'] = min_load_factor
    env_dict['lead_pf_probability'] = lead_pf_probability
    env_dict['load_on_probability'] = load_on_probability

    # Initialize the environment.
    env = gym.make(env_name, **env_dict)

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

    if mod_learn:
        test_loop_mod(env, model)
    else:
        test_loop(env, model)

    # Close the environment (which will flush the log).
    env.close()


def test_loop(env, model):
    for _ in range(5000):
        obs = env.reset()
        done = False

        while not done:
            # env.render()
            obs, rew, done, _ = \
                env.step(model.predict(obs, deterministic=True)[0])


def test_loop_mod(env, model):
    action_list = list()
    for _ in range(5000):
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
         mod_learn):
    """Run the gridmind_reproduce function in a loop."""
    base_dir = os.path.join(THIS_DIR, out_dir)

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
            out_dir=tmp_dir, seed=i, env_name=env_name,
            num_scenarios=num_scenarios, num_time_steps=num_time_steps,
            callback=callback, policy=CustomPolicy, case=case,
            min_load_factor=min_load_factor,
            lead_pf_probability=lead_pf_probability,
            load_on_probability=load_on_probability,
            max_load_factor=max_load_factor, mod_learn=mod_learn
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', help='Relative output directory.',
                        type=str)
    parser.add_argument(
        'env', help='Gym PowerWorld environment to use.', type=str,
        choices=['powerworld-gridmind-env-v0',
                 'powerworld-gridmind-contingencies-env-v0',
                 'powerworld-gridmind-hard-env-v0',
                 'powerworld-discrete-env-simple-14-bus-v0',
                 'powerworld-discrete-env-gen-state-14-bus-v0'
                 ])
    parser.add_argument(
        'case', help='Case to use.', type=str, choices=['14', '14_condensers'])
    parser.add_argument('--num_runs', help='Number of times to train.',
                        type=int, default=5)
    # https://stackoverflow.com/a/24866869/11052174
    parser.add_argument('--hidden_list',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default=[64, 64],
                        help='List of hidden layer sizes, e.g. "64,128"')
    parser.add_argument(
        '--num_scenarios', type=int, default=int(5e5*3),
        help='Number of scenarios for the environment to create.',
    )
    parser.add_argument(
        '--avg_reward', type=float, default=198.75,
        help='Stop training when the 100 episode average has hit this reward.'
    )
    parser.add_argument(
        '--num_time_steps', type=int, default=int(5e5),
        help=('Number of time steps to run training (unless terminated early '
              'by achieving avg_reward). Note that the exploration rate is '
              'currently set to decay linearly from start to num_time_steps.'))
    parser.add_argument(
        '--mod_learn', action='store_true'
    )

    parser.add_argument('--min_load_factor', type=float, default=0.8)
    parser.add_argument('--max_load_factor', type=float, default=1.2)
    parser.add_argument('--load_on_probability', type=float, default=1.0)
    parser.add_argument('--lead_pf_probability', type=float, default=0.0)

    # Parse the arguments.
    args_in = parser.parse_args()

    if args_in.case == '14':
        case_ = IEEE_14_PWB
    elif args_in.case == '14_condensers':
        case_ = IEEE_14_PWB_CONDENSERS
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
         mod_learn=args_in.mod_learn)
