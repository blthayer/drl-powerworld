"""Let's see what voltages look like in the 14 bus case with minimum
and maximum loading.
"""
from gym_powerworld.envs import GridMindEnv
# noinspection PyUnresolvedReferences,PyPackageRequirements
from constants import THIS_DIR, IEEE_14_PWB, IEEE_14_PWB_CONDENSERS, \
    IEEE_14_ONELINE_AXD, IEEE_14_CONTOUR_AXD

# Initialize the environment.
env = GridMindEnv(pwb_path=IEEE_14_PWB, num_scenarios=2,
                  max_load_factor=1.2, min_load_factor=0.8,
                  min_load_pf=None, lead_pf_probability=0,
                  load_on_probability=1, num_gen_voltage_bins=5,
                  gen_voltage_range=(0.95, 1.05),
                  seed=12)

# Grab base case loading.
data = env.load_init_data

# Modify the 1st "episode" to be 80% loading.
env.loads_mw[0, :] = data['LoadSMW'].to_numpy() * 0.8
env.loads_mvar[0, :] = data['LoadSMVR'].to_numpy() * 0.8

# Modify the 2nd "episode" to be 120% loading.
env.loads_mw[1, :] = data['LoadSMW'].to_numpy() * 1.2
env.loads_mvar[1, :] = data['LoadSMVR'].to_numpy() * 1.2

# Call reset to update the case, solve the power flow, and get an
# observation (bus voltages).
obs_low_v = env.reset()
obs_high_v = env.reset()

print(f'80% loading. Min voltage: {obs_low_v.min():.3f}, Max voltage: {obs_low_v.max():.3f}')
print(f'120% loading. Min voltage: {obs_high_v.min():.3fe}, Max voltage: {obs_high_v.max():.3f}')
pass