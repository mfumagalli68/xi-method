import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(type, explain, separation_measurement, **options):
    if type == 'tabular':
        _tabular_plot(explain, separation_measurement, **options)  # k most important variable
    if type == 'image':
        _image_plot()


def _tabular_plot(explain, separation_measurement, **options):

    title = options.get('title', 'Post hoc explainations')
    figsize = options.get('figsize', (10, 10))
    color = options.get('color', 'blue')
    k = options.get('k', 3)

    fig = plt.figure(figsize=figsize)

    try:
        sep = explain.get(separation_measurement)
    except KeyError as e:
        raise KeyError(f"{separation_measurement} not previously computed.")

    data = {'variable': sep.idx_to_col.values(),
            'value': sep.value}

    df = pd.DataFrame(data=data, columns=['variable', 'value'])
    df = df.sort_values('value', ascending=True).reset_index(drop=True)
    if df.shape[0] > k:
        df = df.iloc[:k]

    df = df.sort_values('value', ascending=False).reset_index(drop=True)
    plt.barh('variable', 'value', data=df, color=color)

    plt.title(title)
    plt.show()


def _image_plot(X: np.ndarray):

    plt.imshow(X, cmap='hot', interpolation='nearest')
    plt.show()
