""" Module to test all examples execute run without errors. Run thus:
$ pytest examples/examples.py

Note this run are not automatically tested from GitHub workflows
"""

#-------------------------------------------------------------------------------
import glob
import runpy
import pytest
import matplotlib
matplotlib.use("Qt5Agg")
from pylab import *; ion()

#-------------------------------------------------------------------------------
EXAMPLES_PATH = "examples/*/*.py"
EXAMPLES_LIST = glob.glob(EXAMPLES_PATH)

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("path", EXAMPLES_LIST)
def test_examples(path):
  runpy.run_path(path)
  close('all')

#-------------------------------------------------------------------------------
