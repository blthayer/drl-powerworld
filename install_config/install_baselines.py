"""Helper to install OpenAI's baselines package from source. It is
assumed that the baselines repo exists at the same directory level as
this repository.

https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
"""
import subprocess
import sys
import os

# Uninstall.
subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y',
                       'baselines'])

# Move into the baselines repo.
os.chdir('../../baselines')

# Install.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.'])
