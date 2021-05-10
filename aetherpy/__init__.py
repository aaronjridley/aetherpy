#!/usr/bin/env python
# Copyright 2020, the Aether Development Team (see doc/dev_team.md for members)
# Full license can be found in License.md

# Define a logger object to allow easier log handling
import logging
import os

logging.raiseExceptions = False
logger = logging.getLogger('aetherpy_logger')

# Import the sub-modules
from aetherpy import io, utils, plot

# Define global variables
vfile = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'version.txt')
with open(vfile, 'r') as fin:
    __version__ = fin.read().strip()

# Clean up
del vfile
