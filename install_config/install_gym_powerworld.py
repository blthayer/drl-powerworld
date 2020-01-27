"""Helper to install gym_powerworld from source. It is
assumed that the baselines repo exists at the same directory level as
this repository.

https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
"""
import subprocess
import sys
import os

# Uninstall.
subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y',
                       'gym_powerworld'])

# Move into the baselines repo.
os.chdir('../../gym-powerworld')

# Install.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'])
