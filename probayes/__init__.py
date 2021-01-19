__version__ = '0.0.3.X'

# Probayes namespace
from probayes.constants import NEARLY_POSITIVE_ZERO, \
                               NEARLY_POSITIVE_INF, \
                               NEARLY_NEGATIVE_INF, \
                               LOG_NEARLY_POSITIVE_INF, \
                               COMPLEX_ZERO
from probayes.vtypes import OO
from probayes.named_dict import NamedDict
from probayes.icon import Icon
from probayes.expr import Expr
from probayes.variable import Variable
from probayes.prob import Prob
from probayes.rv import RV
from probayes.field import Field
from probayes.rf import RF
from probayes.sd import SD
from probayes.sp import SP
from probayes.cf import CF
from probayes.manifold import Manifold
from probayes.dist import Dist
from probayes.dist_utils import product, summate, iterdict
from probayes.distribution import Distribution
from probayes.likelihoods import bool_perm_freq
from probayes.expression import Expression
from probayes.sympy_stats import SYMPY_DISTS, sympy_obj_from_dist
from probayes.sympy_prob import SympyProb
