import matplotlib.pyplot as plt
import pandas as pd


def plot(df, type, explain, k=10, **options):
    if type == 'tabular':
        _tabular_plot(df, explain, k=3)  # k most important variable
    if type == 'image':
        _image_plot(df)


def _tabular_plot(df, explain, k, **options):

    title = options.get('title', 'Explainations')
    figsize = options.get('figsize', (10, 10))
    color = options.get('color','blue')

    fig = plt.figure(figsize=figsize)

    data = {'variable': df.columns,
            'value': explain}
    df = pd.DataFrame(data=data, columns=['variable', 'value'])
    df = df.sort_values('value', ascending=True).reset_index(drop=True)
    if df.shape[0]>k:
        df = df.iloc[:k]

    df = df.sort_values('value', ascending=False).reset_index(drop=True)
    plt.barh('variable', 'value', data=df, color=color)

    plt.title(title)
    plt.show()


def _image_plot(df):
    pass
