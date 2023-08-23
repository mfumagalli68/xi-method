![build](https://github.com/mfumagalli68/xi/actions/workflows/python-package.yml/badge.svg)
![last-commit](https://img.shields.io/github/last-commit/mfumagalli68/xi.svg)


<p align="center">
    <em>Xi - Post hoc explainations</em>
</p>

<p align="center">
    <img src="logo.PNG">
</p>
# Xi

**Xi** is a python package that implements the paper "Probabilistic Sensitivity Measures and Classification Tasks".<br>

The growing size and complexity of data as well as the need of accurate predictions, forces analysts to use black-box
model. While the success of those models extends statistical application, it also increases the need for
interpretability and ,when possible, explainability.

The paper propose an approach to the problem based on measures of statistical association.<br>
Measures of statistical association deliver information regarding the strength of the statistical dependence between the
target and the feature(s) of interest, inferring this insight from the data in a **model-agnostic** fashion.<br>
In this respect, we note that an important class of measures of statistical associations is represented by probabilistic
sensitivity measures.<br>

We use these probabilistic sensitivity measures as part of the broad discourse of interpretability in statistical
machine learning. For brevity, we call this part the Xi-method.<br>
Briefly, the method consists in evaluating ML model predictions comparing the values of probabilistic sensitivity
measures obtained in a model-agnostic fashion, i.e., directly from the data, with the same indices computed replacing
the true targets with the ML model forecasts.<br>

# Installation

Install from pypy:

```[python]
pip install xi
```

# Usage

The package is quite simple and it's designed to give you post hoc explainations for your dataset and machine learning
model with minimal effort.<br> 
Import your data in a pandas dataframe format, splitting covariates and independent
variable.<br>

```[python]
df = pd.read_csv("/tests/data/winequality-red.csv", sep=";")
Y = df.quality
df.drop(columns='quality', inplace=True)
```

Create an instance of `XIClassifier` or `XIRegressor` depending on the type of problem you are working with.<br>

```[python]
xi = XIClassifier(m=20)
```

For the classification tasks, you can specify the number of partitions in three different ways:

- `m`: number of partitions can be a dictionary or an integer. The dictionary should have covariate name as key and
  number of desired partition as value. If m is an integer, the desired number of partition will be applied to all
  covariates.
- `discrete`: A list of covariates name you want to treat as categorical.
- `obs`: A dictionary mapping covariates name to number of desired observations in each partition.

A default `m` value will be computed if nothing is provided by the user, as indicated in the paper.<br>

To obtain post hoc explanations simply run:

```[python]
p = xi.explain(X=df, y=Y, separation_measurement='L1')
```

Object `p` will contain explanation value and an index to covariate name mapping, helpful to associate explanation with
the corresponding covariate.<br>

You can choose from different separation measurement, as specified in the paper. You can specify one separation
measurement or more than one, using a list.

```[python]
p = xi.explain(X=df, y=Y, separation_measurement=['L1','Kuiper'])
```

Implemented separation measurement can be viewed running:

```[python]
get_separation_measurement()
```

Plot you result:

```[python]
plot(separation_measurement='L1', type='tabular', explain=P, k=10)
```