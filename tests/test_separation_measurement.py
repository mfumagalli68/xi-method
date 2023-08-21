from xi.ximp import *
from xi.utils import check_args_overlap
from xi.exceptions import XiError
import pytest
from pathlib import Path
import numpy as np
def test_separation_measurement():

    np.random.seed(3)
    df = pd.read_csv(Path('data', 'winequality-red.csv'), sep=";")
    Y = df.quality.values
    df.drop(columns='quality', inplace=True)

    xi = XIClassifier(m=20)
    P = xi.explain(X=df, y=Y, separation_measure='L1',replicates=50)

    exp = np.array([0.2109063,0.19921303,0.1841079,0.15099798,0.12599739,0.20654458, 0.28604079,0.16488397,0.16463576,0.1880844,0.42336663])
    np.testing.assert_allclose(P.get('L1').value,
                               exp,
                               rtol=0.05,
                               atol=0.05)
