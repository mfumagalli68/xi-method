import  numpy as np
from xi_method.ximp import XIClassifier
import pandas as pd
import os

path = 'tests/data/winequality-red.csv'
def test_plot():

    k=3
    np.random.seed(3)
    df = pd.read_csv(os.path.abspath(path), sep=";")
    Y = df.quality.values
    df.drop(columns='quality', inplace=True)


    xi = XIClassifier(m=20)
    p = xi.explain(X=df, y=Y, replicates=50, separation_measurement='L1')
    #try:
    sep = p.get('L1')

    data = {'variable': sep.idx_to_col.values(),
            'value': sep.explanation}
    df = pd.DataFrame(data=data, columns=['variable', 'value'])

    df = df.sort_values('value', ascending=True).reset_index(drop=True)
    if df.shape[0] > k:
        df = df.iloc[:k]

    df = df.sort_values('value', ascending=False).reset_index(drop=True)

    pd.testing.assert_series_equal(df.variable,
                                   pd.Series(['chlorides','pH','density'],
                                             name='variable'))