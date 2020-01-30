import os
# Disable GPU.
# os.environ['CUDA_VISIBLE_DEVICES'] = ""
import gym
# Must import gym_powerworld for the environments to get registered.
# noinspection PyUnresolvedReferences
import gym_powerworld
# noinspection PyPackageRequirements
from baselines import deepq
import logging
import numpy as np
import time
import tensorflow as tf
import shutil
from copy import deepcopy

# noinspection PyUnresolvedReferences,PyPackageRequirements
from constants import THIS_DIR, IEEE_14_PWB, IEEE_14_PWB_CONDENSERS, \
    IEEE_14_ONELINE_AXD, IEEE_14_CONTOUR_AXD

# Dictionary of GridMind environment inputs.
GRIDMIND_DICT = dict(
    pwb_path=IEEE_14_PWB_CONDENSERS,
    # Create 20000 scenarios, though we likely won't use
    # them all.
    num_scenarios=20000,
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
    # image_dir=image_dir,
    # csv_logfile=train_logfile,
)

BASELINES_DICT = dict(
    network='mlp',
    # Use default learning rate.
    lr=5e-4,
    # Would like to run for 10,000 episodes. We'll set the total
    # time steps to a large number and use the callback to terminate
    # training.
    total_timesteps=100000,
    # Defaults:
    buffer_size=50000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    train_freq=1,
    batch_size=32,
    print_freq=100,
    checkpoint_path=None,
    learning_starts=1000,
    gamma=1.0,
    target_network_update_freq=500,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta0=0.4,
    prioritized_replay_beta_iters=None,
    prioritized_replay_eps=1e-6,
    param_noise=False,
    load_path=None,
    # Don't checkpoint - use our callback to stop training.
    checkpoint_freq=None,
    # Use prioritized replay.
    prioritized_replay=True,
    # Modify inputs to the neural network. Since the output
    # layer is large (5^5 = 3125), the default 64 node layers
    # are probably a bit small. Also, why is tanh the default?
    num_layers=2, num_hidden=128, activation=tf.nn.relu,
    # Update the following:
    # seed=seed,
    # env=env,
    # callback=gridmind_callback,
)


def gridmind_callback(lcl, _glb) -> bool:
    """
    Stop training if the agent is "one-shotting" 99 of 100 episodes,
    and "two-shotting" the remaining 1.

    A single action episode will get a reward of 200 (100 for all
    voltages being in bounds plus an end of episode reward of 100).

    For a two action episode, assume after the first action, there
    is at least one bus in the "violation" zone and none in the
    "diverged" zone. This results in a "reward" of -50. Assume the
    second action puts all buses in the good zone, and thus gets a
    reward of 100. The end of episode reward is then (100 - 50) / 2
    = 25. The total episode reward is then 75.

    So, the average should be (99 * 200) + (1 * 75) / 100 = 198.75.

    Alo stop training if we've hit 10,000 episodes, as that's what the
    GridMind team trained to. We're using a different neural net, so
    this may or may not be effective.

    :param lcl: locals() inside deepq.learn
    :param _glb: globals() inside deepq.learn
    """
    # Compute the average of the last 100 episodes.
    avg_100 = sum(lcl['episode_rewards'][-101:-1]) / 100

    # The length of 'episode_rewards' indicates how many episodes we've
    # gone through.
    num_ep = len(lcl['episode_rewards'])

    if (avg_100 >= 198.75) or (num_ep >= 10000):
        # Terminate training.
        print('Terminating training since either the 100 episode average '
              'reward has exceeded 197.5 or we have exceeded 10,000 episodes.')
        return True
    else:
        # Don't terminate training.
        return False


def gridmind_reproduce(out_dir, seed):
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
    input_dict = deepcopy(GRIDMIND_DICT)

    # overwrite the seed, image_dir, and csv_logfile.
    input_dict['seed'] = seed
    input_dict['image_dir'] = image_dir
    input_dict['csv_logfile'] = train_logfile

    # Initialize the environment.
    env = gym.make('powerworld-gridmind-env-v0', **input_dict)

    # Get a copy of the default inputs for deepq.
    learn_dict = deepcopy(BASELINES_DICT)

    # Overwrite seed, env, and callback.
    learn_dict['seed'] = seed
    learn_dict['env'] = env
    learn_dict['callback'] = gridmind_callback

    # Learning time.
    t0 = time.time()
    # noinspection PyTypeChecker
    act = deepq.learn(**learn_dict)
    t1 = time.time()

    print('All done, saving to file.')
    act.save(model_file)

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
            obs, rew, done, _ = env.step(
                act(obs[None], stochastic=False, update_eps=-1)[0])

        # Render again at the end.
        # env.render()

    # Close the environment (which will flush the log).
    env.close()


def gridmind_reproduce_loop(runs):
    """Run the gridmind_reproduce function in a loop."""
    base_dir = os.path.join(THIS_DIR, 'gridmind_reproduce')

    for i in range(runs):
        tmp_dir = os.path.join(base_dir, f'run_{i}')

        try:
            os.mkdir(tmp_dir)
        except FileExistsError:
            # Delete the directory if it's present, and remake it.
            shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)

        gridmind_reproduce(out_dir=tmp_dir, seed=i)


if __name__ == '__main__':
    gridmind_reproduce_loop(10)
