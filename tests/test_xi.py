import os.path
from xi.ximp import *
import numpy as np

path = 'tests/data/winequality-red.csv'


def test_separation_measurement_m_int():
    np.random.seed(3)
    df = pd.read_csv(os.path.abspath(path), sep=";")
    Y = df.quality.values
    df.drop(columns='quality', inplace=True)

    xi = XIClassifier(m=20)
    p = xi.explain(X=df, y=Y, replicates=50, separation_measurement='L1')

    exp = np.array(
        [0.2109063, 0.19921303, 0.1841079, 0.15099798, 0.12599739, 0.20654458, 0.28604079, 0.16488397, 0.16463576,
         0.1880844, 0.42336663])
    np.testing.assert_allclose(p.get('L1').explanation,
                               exp,
                               rtol=0.05,
                               atol=0.05)


def test_separation_measurement_obs():
    np.random.seed(3)
    df = pd.read_csv(os.path.abspath(path), sep=";")
    Y = df.quality.values
    df.drop(columns='quality', inplace=True)

    xi = XIClassifier(obs={'fixed acidity': 100})
    p = xi.explain(X=df, y=Y, replicates=1, separation_measurement='L1')

    assert isinstance(p.get('L1').explanation, np.ndarray)


def test_separation_measurement_discrete():
    np.random.seed(3)
    df = pd.read_csv(os.path.abspath(path), sep=";")
    Y = df.quality.values
    df.drop(columns='quality', inplace=True)

    xi = XIClassifier(discrete=['fixed acidity'])
    p = xi.explain(X=df, y=Y, replicates=1, separation_measurement='L1')

    assert isinstance(p.get('L1').explanation, np.ndarray)


def test_separation_measurement_m_dict():
    np.random.seed(3)
    df = pd.read_csv(os.path.abspath(path), sep=";")
    Y = df.quality.values
    df.drop(columns='quality', inplace=True)

    xi = XIClassifier(m={'fixed acidity': 100})
    p = xi.explain(X=df, y=Y, replicates=1, separation_measurement='L1')

    assert isinstance(p.get('L1').explanation, np.ndarray)


def test_separation_measurement_regressor():
    np.random.seed(2)
    Y = np.random.normal(size=1000)
    X = np.random.normal(3, 2, size=5 * 1000)  # df_np[:, 1:11]
    X = X.reshape((1000, 5))

    xi = XIRegressor(m=10, grid=100)
    p = xi.explain(X=X, y=Y, replicates=1,
                   separation_measurement='Kullback-leibler')

    exp = np.array(
        [0.065, 0.05, 0.065, 0.061, 0.05])
    np.testing.assert_allclose(p.get('Kullback-leibler').explanation,
                               exp,
                               rtol=0.05,
                               atol=0.05)


def test_separation_measurement_hellinger():
    np.random.seed(3)
    df = pd.read_csv(os.path.abspath(path), sep=";")
    Y = df.quality.values
    df.drop(columns='quality', inplace=True)

    xi = XIClassifier(m=20)
    p = xi.explain(X=df, y=Y, replicates=1, separation_measurement='Hellinger')

    exp = np.array(
        [0.01293297, 0.01992071, 0.01205301, 0.01015738, 0.01004335, 0.01540372, 0.02567093, 0.01070475, 0.0121346,
         0.01461179,
         0.04954897])
    np.testing.assert_allclose(p.get('Hellinger').explanation,
                               exp,
                               rtol=0.05,
                               atol=0.05)


def test_separation_measurement_L2():
    np.random.seed(3)
    df = pd.read_csv(os.path.abspath(path), sep=";")
    Y = df.quality.values
    df.drop(columns='quality', inplace=True)

    xi = XIClassifier(m=20)
    p = xi.explain(X=df, y=Y, replicates=1, separation_measurement='L2')

    exp = np.array(
        [0.01293297, 0.01992071, 0.01205301, 0.01015738, 0.01004335, 0.01540372, 0.02567093, 0.001070475, 0.00821346,
         0.01461179,
         0.04954897])
    np.testing.assert_allclose(p.get('L2').explanation,
                               exp,
                               rtol=0.05,
                               atol=0.05)
