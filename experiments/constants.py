import os
import logging
import numpy as np

# Get this directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'cases'))
IEEE_14_DIR = os.path.join(CASE_DIR, 'ieee_14')
IL_200_DIR = os.path.join(CASE_DIR, 'il_200')
SC_500_DIR = os.path.join(CASE_DIR, 'sc_500')

# Constants for IEEE 14 bus files.
IEEE_14_PWB = os.path.join(IEEE_14_DIR, 'IEEE 14 bus.pwb')
IEEE_14_PWB_CONDENSERS = os.path.join(IEEE_14_DIR,
                                      'IEEE 14 bus condensers.PWB')
IEEE_14_ONELINE_AXD = os.path.join(IEEE_14_DIR, 'IEEE 14 bus.axd')
IEEE_14_CONTOUR_AXD = os.path.join(IEEE_14_DIR, 'contour.axd')

IL_200_PWB = os.path.join(IL_200_DIR, 'ACTIVSg200.pwb')
SC_500_PWB = os.path.join(SC_500_DIR, 'ACTIVSg500.pwb')

# Write to a faster SSD to make life better.
DATA_DIR = r'D:\\runs'

# Defines some defaults.
MIN_LOAD_FACTOR_DEFAULT = 0.6
MAX_LOAD_FACTOR_DEFAULT = 1.4
MIN_LOAD_PF_DEFAULT = 0.8
LOAD_ON_PROBABILITY_DEFAULT = 0.9
LEAD_PF_PROBABILITY_DEFAULT = 0.1
NUM_TIME_STEPS_DEFAULT = int(5e5)
SHUNT_CLOSED_PROBABILITY_DEFAULT = 0.5
NUM_GEN_VOLTAGE_BINS_DEFAULT = 5
GEN_VOLTAGE_RANGE_DEFAULT = (0.95, 1.05)
LOW_V_DEFAULT = 0.95
HIGH_V_DEFAULT = 1.05
TRUNCATE_VOLTAGES_DEFAULT = False
# Make plenty of scenarios so we never run out.
NUM_SCENARIOS_DEFAULT = int(NUM_TIME_STEPS_DEFAULT * 4)
NUM_RUNS_DEFAULT = 3

SCREEN_DIR_14 = os.path.join(THIS_DIR, 'screen_14')

# How much testing we're doing.
TEST_EPISODES = 5000

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
    shunt_closed_probability=0.5,
    min_load_pf=MIN_LOAD_PF_DEFAULT,
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
    # truncate_voltages=False
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
    # Have exploration linearly decay based on total_timesteps.
    exploration_fraction=1.0,
    # Update the following:
    # gamma=1.0,
    # policy=policy
    # seed=seed
    # env=env
)


def get_file_str(case_str, seed, contingencies):
    return f'_{case_str}_{seed}_{int(contingencies)}'


def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.

    Source: https://stackoverflow.com/a/48372659/11052174
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with two decimal places
        label = "{:.2f}".format(y_value)

        # Create annotation
        ax.annotate(
            # Use `label` as label
            label,
            # Place label at end of the bar
            (x_value, y_value),
            # Place label at end of the bar
            xytext=(0, space),
            # Interpret `xytext` as offset in points
            textcoords="offset points",
            # Horizontally center label
            ha='center',
            # Vertically align label differently for positive and
            # negative values.
            va=va)
