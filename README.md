# eptune
> eptune (evolutionary parameter tuning) is a python package trying to use evolutionary computation algorithms to do parameter tuning.


![CI](https://github.com/wanglongqi/eptune/workflows/CI/badge.svg)

![Logo](nbs/eptune.png)

## Install

`pip install eptune`

## How to use

Using following lines can fine tune MNIST dataset with 4-Fold CV performance using the `qtuneSimple` function.

```python
from eptune.sample_cases import DigitsCV
from eptune.quick import qtuneSimple
from eptune.parameter import *
from sklearn.svm import SVC

# Prameter space to search
params = [
    LogFloatParameter([0.01, 1e4], 'C'),
    CategoricalParameter(['rbf'], 'kernel'),
    LogFloatParameter([1e-6, 1e4], 'gamma')
]

# Define objective function
cv_svc_digits = DigitsCV(SVC())


def evaluate(params):
    return cv_svc_digits.cv_loss_with_params(callbacks=tuple(), **params)


# Call `qtuneSimple`
population, logbook, hof = qtuneSimple(params,
                                       evaluate,
                                       n_pop=10,
                                       n_jobs=10,
                                       mutpb=0.6,
                                       cxpb=0.8,
                                       seed=42)

# Plot the logbook if needed
fig = logbook.plot(['min', 'avg'])
```

    gen	nevals	avg          	std        	min          	max          
    0  	10    	[-0.28174736]	[0.3288165]	[-0.96772398]	[-0.10072343]
    1  	7     	[-0.70684474]	[0.36593114]	[-0.97273233]	[-0.10072343]
    2  	4     	[-0.8786867] 	[0.2590384] 	[-0.97273233]	[-0.10183639]
    3  	8     	[-0.62526433]	[0.41696083]	[-0.97440178]	[-0.10072343]
    4  	8     	[-0.80116861]	[0.34319099]	[-0.97440178]	[-0.10072343]
    5  	6     	[-0.96143573]	[0.0257779] 	[-0.97440178]	[-0.89816361]
    6  	7     	[-0.9475793] 	[0.06357501]	[-0.97440178]	[-0.75959933]
    7  	6     	[-0.97250974]	[0.00531551]	[-0.97440178]	[-0.95659432]
    8  	7     	[-0.97445743]	[0.00016694]	[-0.97495826]	[-0.97440178]
    9  	8     	[-0.73567056]	[0.36697176]	[-0.97495826]	[-0.10072343]
    10 	7     	[-0.79810796]	[0.34639554]	[-0.97495826]	[-0.10072343]



![png](docs/images/output_6_1.png)


The best parameters are stored in `HallofFame` object:

```python
hof
```




    [({'C': 197.75053974020003, 'kernel': 'rbf', 'gamma': 0.0005362324820364681}, (-0.9749582637729549,)), ({'C': 197.75053974020003, 'kernel': 'rbf', 'gamma': 0.00044545277111534496}, (-0.9744017807456873,))]



## More control

If you want more control, you can check:
1. `eptune.sklearn` module provides `ScikitLearner` or `ScikitLearnerCV` for fine tune parameter of estimators with scikit learn API. Examples are also provided in the documentation.
2. `eptune.algorithms` module provides algorithms to access the DEAP framework directly.
