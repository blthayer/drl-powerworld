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
from stable_baselines.deepq.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines import DQN
import argparse

# noinspection PyUnresolvedReferences,PyPackageRequirements
from constants import THIS_DIR, IEEE_14_PWB, IEEE_14_PWB_CONDENSERS, \
    IEEE_14_ONELINE_AXD, IEEE_14_CONTOUR_AXD

# Dictionary of GridMind environment inputs.
GRIDMIND_DICT = dict(
    pwb_path=IEEE_14_PWB_CONDENSERS,
    # GridMind team did loading from 80% to 120%
    max_load_factor=1.2, min_load_factor=0.8,
    # All loads were always on, and no power factors were
    # changed.
    lead_pf_probability=None, load_on_probability=None,
    # Five voltage bins: [0.95, 0.975, 1.0, 1.025, 1.05]
    num_gen_voltage_bins=5,
    gen_voltage_range=(0.95, 1.05),
    log_level=logging.INFO,
    # Use the same reward values.
    rewards=dict(normal=100, violation=-50, diverged=-100),
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
    # seed=seed,
    # num_scenarios=num_scenarios
    # image_dir=image_dir,
    # csv_logfile=train_logfile,
)

BASELINES_DICT = dict(
    # The following are all defaults:
    gamma=0.99,
    learning_rate=5e-4,
    buffer_size=50000,
    exploration_final_eps=0.02,
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
    # Have exploration linearly decay based on total_timesteps.
    exploration_fraction=1.0,
    # Update the following:
    # policy=policy
    # seed=seed
    # env=env
)


def callback_factory(max_episodes, average_reward):

    def callback(lcl, _glb) -> bool:
        """
        :param lcl: locals() inside deepq.learn
        :param _glb: globals() inside deepq.learn
        """
        # Compute the average of the last 100 episodes.
        avg_100 = sum(lcl['episode_rewards'][-101:-1]) / 100

        # The length of 'episode_rewards' indicates how many episodes we've
        # gone through.
        num_ep = len(lcl['episode_rewards'])

        if (avg_100 >= average_reward) or (num_ep >= max_episodes):
            # Terminate training.
            print('Terminating training since either the 100 episode average '
                  f'reward has exceeded {average_reward} or we have exceeded '
                  f'{max_episodes} episodes.')
            return False
        else:
            # Don't terminate training.
            return True

    return callback


def learn_and_test(out_dir, seed, env_name, num_scenarios, num_time_steps,
                   callback, policy):
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
    env_dict = deepcopy(GRIDMIND_DICT)

    # overwrite the seed, image_dir, and csv_logfile.
    env_dict['seed'] = seed
    env_dict['image_dir'] = image_dir
    env_dict['csv_logfile'] = train_logfile
    env_dict['num_scenarios'] = num_scenarios

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

    for _ in range(2000):
        obs = env.reset()
        done = False

        while not done:
            # env.render()
            obs, rew, done, _ = env.step(model.predict(obs)[0])

        # Render again at the end.
        # env.render()

    # Close the environment (which will flush the log).
    env.close()


def loop(out_dir, env_name, runs, hidden_list, num_scenarios,
         max_episodes, avg_reward, num_time_steps):
    """Run the gridmind_reproduce function in a loop."""
    base_dir = os.path.join(THIS_DIR, out_dir)

    # Make the directory. Don't worry if it exists already.
    try:
        os.mkdir(base_dir)
    except FileExistsError:
        pass

    # Create the callback.
    callback = callback_factory(max_episodes=max_episodes,
                                average_reward=avg_reward)

    # Create a custom policy using the specified layers.
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            # noinspection PyUnresolvedReferences
            super(CustomPolicy, self).__init__(
                *args, **kwargs, layers=hidden_list, layer_norm=False,
                feature_extraction='mlp', act_fun=tf.nn.relu)

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
            callback=callback, policy=CustomPolicy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', help='Relative output directory.',
                        type=str)
    parser.add_argument(
        'env', help='Gym PowerWorld environment to use.', type=str,
        choices=['powerworld-gridmind-env-v0',
                 'powerworld-gridmind-contingencies-env-v0'])
    parser.add_argument('--num_runs', help='Number of times to train.',
                        type=int, default=1)
    # https://stackoverflow.com/a/24866869/11052174
    parser.add_argument('--hidden_list',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default=[64, 128],
                        help='List of hidden layer sizes, e.g. "64,128"')
    parser.add_argument(
        '--num_scenarios', type=int, default=50000,
        help='Number of scenarios for the environment to create.',
    )
    parser.add_argument(
        '--max_episodes', type=int, default=45000,
        help='Maximum number of training episodes to run before stopping.')
    parser.add_argument(
        '--avg_reward', type=float, default=198.75,
        help='Stop training when the 100 episode average has hit this reward.'
    )
    parser.add_argument(
        '--num_time_steps', type=int, default=100000,
        help=('Number of time steps to run training (unless terminated early '
              'by hitting max_episodes or achieving avg_reward). Note that the'
              ' exploration rate is currently set to decay linearly from '
              'start to num_time_steps.'))

    # Parse the arguments.
    args_in = parser.parse_args()

    # Run.
    loop(out_dir=args_in.out_dir, env_name=args_in.env, runs=args_in.num_runs,
         hidden_list=args_in.hidden_list, num_scenarios=args_in.num_scenarios,
         max_episodes=args_in.max_episodes, avg_reward=args_in.avg_reward,
         num_time_steps=args_in.num_time_steps)
