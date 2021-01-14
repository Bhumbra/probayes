# Module to test Expr

#-------------------------------------------------------------------------------
import glob
import pytest

#-------------------------------------------------------------------------------
EXAMPLES_PATH = "examples/*/*.py"
EXAMPLES_LIST = glob.glob(EXAMPLES_PATH)

#-------------------------------------------------------------------------------
@pytest.mark.parametrize("path", EXAMPLES_LIST)
def test_examples(path):
  with open(path, 'r') as readpy:
    exec(readpy.read())

#-------------------------------------------------------------------------------
