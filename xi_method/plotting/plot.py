import sys

import matplotlib.pyplot as plt
import pandas as pd
from typing import *
from xi_method.separation.measurement import SeparationMeasurement
from xi_method.exceptions import XiError

def plot(type: AnyStr, explain: Dict, separation_measurement: AnyStr, **options):
    if type in ('tabular','text'):
        _tabular_plot(explain, separation_measurement, **options)  # k most important variable
    if type == 'image':
        _image_plot(explain, separation_measurement, **options)


def _tabular_plot(explain: Dict, separation_measurement: AnyStr, **options):
    title = options.get('title', 'Explanations')
    figsize = options.get('figsize', (10, 10))
    color = options.get('color', 'blue')
    k = options.get('k', 3)

    fig = plt.figure(figsize=figsize)

    try:
        sep = explain.get(separation_measurement)
    except KeyError as e:
        raise KeyError(f"{separation_measurement} not previously computed.")

    data = {'variable': sep.idx_to_col.values(),
            'value': sep.explanation}

    df = pd.DataFrame(data=data, columns=['variable', 'value'])
    df = df.sort_values('value', ascending=True).reset_index(drop=True)
    if df.shape[0] > k:
        df = df.iloc[:k]

    df = df.sort_values('value', ascending=False).reset_index(drop=True)
    plt.barh('variable', 'value', data=df, color=color)

    plt.title(title)
    plt.show()


def _image_plot(explain: Dict,
                separation_measurement: AnyStr,
                shape: tuple,
                **options):

    sep = explain.get(separation_measurement).explanation
    try:
        sep = sep.reshape(shape)
    except ValueError as e:
        raise XiError(e)
    plt.imshow(sep, cmap='hot', interpolation='nearest')
    plt.show()
