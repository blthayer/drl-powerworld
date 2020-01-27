import os

# Get this directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'cases'))
IEEE_14_DIR = os.path.join(CASE_DIR, 'ieee_14')

# Constants for IEEE 14 bus files.
IEEE_14_PWB = os.path.join(IEEE_14_DIR, 'IEEE 14 bus.pwb')
IEEE_14_PWB_CONDENSERS = os.path.join(IEEE_14_DIR,
                                      'IEEE 14 bus condensers.PWB')
IEEE_14_ONELINE_AXD = os.path.join(IEEE_14_DIR, 'IEEE 14 bus.axd')
IEEE_14_CONTOUR_AXD = os.path.join(IEEE_14_DIR, 'contour.axd')