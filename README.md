![python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![build](https://github.com/mfumagalli68/xi/actions/workflows/python-package.yml/badge.svg)
# XI

XI is a python package that implements the paper "Probabilistic Sensitivity Measures and
Classification Tasks".

# Getting started


# Installation
Install from pypy:

```[python]
pip install xi
```

# Usage

The package is quite simple and it's designed to give you
post hoc explainations for your dataset and machine learning model
with minimal effort.
Import your data in a pandas dataframe format and your predictions
in a numpy array format.<br>
Create an instance of `XIClassifier` or `XIRegressor` depending on the type of 
problem you are working with.<br>

```[python]
xi = XIClassifier(m=20)
```
For the classification tasks, you can specify the number of partitions in three different ways:

- *m*: number of partitions can be a dictionary or an integer. The dictionary should have covariate name
as key and number of desired partition as value. If m is an integer, the desired number of partition will be applied
to all covariates.
- *discrete*: A list of covariates name you want to treat as categorical.
- *obs*: A dictionary mapping covariates name to number of desired observations in each partition.

A default *m* value will be computed if nothing is provided by the user, as indicated in the paper.<br>

To obtain post hoc explainations simpy run:

```[python]
p = xi.explain(X=df, y=Y, separation_measurement='L1')
```
Object `p` will contain explanation value and an index to covariate name mapping, 
helpful to associate explaination with the corresponding covariate.<br>

You can choose from different separation measurement, as specified in the paper.
You can specify one separation measurement or more than one, using a list.

```[python]
p = xi.explain(X=df, y=Y, separation_measurement=['L1','Kuiper'])
```

Implemented separation measurement can be viewed running:

```[python]
TBD
```

