import gym
# Must import gym_powerworld for the environments to get registered.
import gym_powerworld
from baselines import deepq

# Temporarily hard code path to 14 bus case.
# TODO: fix this
PATH_14 = \
    r'C:\Users\blthayer\git\gym-powerworld\tests\cases\ieee_14\IEEE 14 bus.pwb'

#
def main():
    env = gym.make('powerworld-gridmind-env-v0', pwb_path=PATH_14,
                   num_scenarios=20000, max_load_factor=1.2,
                   min_load_factor=0.8, lead_pf_probability=None,
                   load_on_probability=None, num_gen_voltage_bins=5,
                   gen_voltage_range=(0.95, 1.05), seed=42)
    act = deepq.learn(
        env=env,
        network='mlp'
    )

if __name__ == '__main__':
    main()