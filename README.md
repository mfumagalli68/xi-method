![build](https://github.com/mfumagalli68/xi/actions/workflows/build.yml/badge.svg)
![coverage](https://codecov.io/gh/mfumagalli68/xi/branch/main/graph/badge.svg)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)


<p align="center">
    <em>Xi - Post hoc explanations</em>
</p>

<p align="center">
    <img src="logo.PNG">
</p>


**Xi** is a python package that implements the paper 
"Explaining classifiers with measures of statistical association"[[1]](#1)
and "The Xi method: unlocking the mysteries of regression with Statistics"[[2]](#2)<br>

The growing size and complexity of data as well as the need of accurate predictions, forces analysts to use black-box
model. While the success of those models extends statistical application, it also increases the need for
interpretability and ,when possible, explainability.

The paper proposes an approach to the problem based on measures of statistical association.<br>
Measures of statistical association deliver information regarding the strength of the statistical dependence between the
target and the feature(s) of interest, inferring this insight from the data in a **model-agnostic** fashion.<br>
In this respect, we note that an important class of measures of statistical associations is represented by probabilistic
sensitivity measures.<br>

We use these probabilistic sensitivity measures as part of the broad discourse of interpretability in statistical
machine learning. For brevity, we call this part the Xi-method.<br>
Briefly, the method consists in evaluating ML model predictions comparing the values of probabilistic sensitivity
measures obtained in a model-agnostic fashion, i.e., directly from the data, with the same indices computed replacing
the true targets with the ML model forecasts.<br>

To sum up, **Xi** has three main advantages:

- Model agnostic: as long as your model outputs predictions, you can use **Xi** with any model
- Data agnostic: **Xi** works with structured (tabular) and unstructured data ( text, image ).
- Computationally reasonable

# Installation

Install from pypi:

```[python]
pip install xi-method
```

# Usage

The package is quite simple and it's designed to give you post hoc explainations for your dataset and machine learning
model with minimal effort.<br> 
Import your data in a pandas dataframe format, splitting covariates and independent
variable.<br>

```[python]
from xi_method.utils import load_dataset
from xi_method.ximp import *

# load wine quality
df = load_wine_quality_red_dataset()

Y = df.quality
df.drop(columns='quality', inplace=True)
```

Create an instance of `XIClassifier` or `XIRegressor` depending on the type of problem you are working with:<br>

```[python]
xi = XIClassifier(m=20)
```

For the classification tasks, you can specify the number of partitions in three different ways:

- `m`: number of partitions can be a dictionary or an integer. The dictionary should have covariate name as key and
  number of desired partition as value. If m is an integer, the desired number of partition will be applied to all
  covariates.
- `discrete`: A list of covariates name you want to treat as categorical.
- `obs`: A dictionary mapping covariates name to number of desired observations in each partition.

For regression tasks, you can only specify `m` as an integer.<br>

A default `m` value will be computed if nothing is provided by the user, as indicated in the paper.<br>

To obtain post hoc explanations, run your favorite ML model, save the predictions as numpy array
and provide the covariates ( test set) and the predictions to the method `explain`:

```[python]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.values, Y, test_size=0.3, random_state=42)
lr = LogisticRegression(multi_class='multinomial',max_iter=100)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

xi = XIClassifier(m=20)
p = xi.explain(X=x_test, y=y_pred, replicates=10, separation_measurement='L1')
```

Object `p` is a python dictionary mapping separation measurement and explanation.<br>
You can easily have access to the explanation:

```[python]
p.get('L1').explanation
```

You can choose from different separation measurement, as specified in the paper. You can specify one separation
measurement or more than one, using a list.

```[python]
p = xi.explain(X=x_test, y=y_pred, separation_measurement=['L1','Kuiper'])
```

Implemented separation measurement:

- Kullback - Leibler
- Kuiper
- L1
- L2
- Hellinger

You can get a list of implemented separation measurement running:

```[python]
from xi_method.utils import *
get_separation_measurement()
```

Plot your result:

```[python]
plot(separation_measurement='L1', type='tabular', explain=P, k=10)
```


## References
<a id="1">[1]</a> 
E. Borgonovo, V. Ghidini, R. Hahn a, E. Plischke (2023). 
Explaining classifiers with measures of statistical association
Computational Statistics and Data Analysis, Volume 182, June 2023, 107701

<a id="2">[2]</a> 
V. Ghidini (2023). 
The Xi method: unlocking the mysteries of regression with Statistics
