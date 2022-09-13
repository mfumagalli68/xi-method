from xi.utils import check_args_overlap
from xi.exceptions import XiError

def test_overlap_arg(self):
    m = {'1':1,'2':2}
    obs = {'1': 1, '4': 2, '3': 1}
    discrete = {'1': 1, '2': 2}

    self.assertRaises(XiError, check_args_overlap(m,obs,discrete))