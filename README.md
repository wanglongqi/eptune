# eptune
> eptune (evolutionary parameter tuning) is a python package trying to use evolutionary computation algorithms to do parameter tuning.


## Install

`pip install eptune`

## How to use

### Optimize parameters based on the performance on validation dataset

Following is a example to use the customized algorithms from `ept.algorithms`: 

```python
from eptune.sample_cases import svc_digits, svc_digits_proba
from eptune.algorithms import eaSimpleWithExtraLog, eaMuPlusLambdaWithExtraLog, eaMuCommaLambdaWithExtraLog
from eptune.parameter import *

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from functools import partial
from sklearn.metrics import *

toolbox = base.Toolbox()
params = [
    LogFloatParameter([0.1, 1000], 'C'),
    CategoricalParameter(['poly', 'rbf', 'sigmoid'], "kernel"),
    LogFloatParameter([1e-6, 1e6], 'gamma')
]


def initParams(cls):
    return cls({i.name: next(i) for i in cls.params})


def evaluate(params):
    return svc_digits.val_loss_with_params(callbacks=(accuracy_score, ), **params)


creator.create("Loss", base.Fitness, weights=(-1.0, ))
creator.create("Parameters", dict, params=params, fitness=creator.Loss)
toolbox.register("individual", initParams, creator.Parameters)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate)
from ept.utils import ConcurrentMap
pmap = ConcurrentMap(10)
toolbox.register('map', pmap.map)

toolbox.register("select", tools.selTournament, tournsize=3)

from ept.crossover import cxDictUniform
toolbox.register("mate", cxDictUniform, indpb=0.6)

from ept.mutation import mutDictRand
toolbox.register("mutate", partial(mutDictRand, params=params, indpb=0.75))

import numpy
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean, axis=0)
stats.register("std", numpy.std, axis=0)
stats.register("min", numpy.min, axis=0)
stats.register("max", numpy.max, axis=0)
hof = tools.HallOfFame(2)


def run():
    return eaSimpleWithExtraLog(toolbox.population(10),
                                toolbox,
                                cxpb=0.6,
                                mutpb=0.6,
                                ngen=16,
                                halloffame=hof,
                                elitism=True,
                                stats=stats)
```

```python
%time population, logbook = run()
```

    gen	nevals	avg          	std         	min          	max          
    0  	10    	[-0.77083333]	[0.36533481]	[-0.99444444]	[-0.06944444]
    1  	8     	[-0.71694444]	[0.42388825]	[-0.99444444]	[-0.06944444]
    2  	5     	[-0.71666667]	[0.42370718]	[-0.99444444]	[-0.06944444]
    3  	7     	[-0.89444444]	[0.27590759]	[-0.99444444]	[-0.06944444]
    4  	8     	[-0.71611111]	[0.42334974]	[-0.99444444]	[-0.06944444]
    5  	7     	[-0.78777778]	[0.36184797]	[-0.99722222]	[-0.06944444]
    6  	7     	[-0.8075]    	[0.36907469]	[-0.99722222]	[-0.06944444]
    7  	5     	[-0.9025]    	[0.27768735]	[-0.99722222]	[-0.06944444]
    8  	8     	[-0.90027778]	[0.2770308] 	[-0.99722222]	[-0.06944444]
    9  	4     	[-0.7175]    	[0.42425325]	[-0.99722222]	[-0.06944444]
    10 	6     	[-0.81027778]	[0.37041862]	[-0.99722222]	[-0.06944444]
    11 	7     	[-0.8275]    	[0.29185298]	[-0.99722222]	[-0.06944444]
    12 	6     	[-0.8775]    	[0.26790641]	[-0.99722222]	[-0.125]     
    13 	8     	[-0.97472222]	[0.06209374]	[-1.]        	[-0.78888889]
    14 	6     	[-0.90416667]	[0.27826138]	[-1.]        	[-0.06944444]
    15 	5     	[-0.99527778]	[0.00703277]	[-1.]        	[-0.98055556]
    16 	8     	[-0.61944444]	[0.44953165]	[-1.]        	[-0.06944444]
    CPU times: user 139 ms, sys: 49 ms, total: 188 ms
    Wall time: 8.07 s


```python
population
```




    [{'C': 108.33526825316423,
      'kernel': 'sigmoid',
      'gamma': 0.00022578116889877787},
     {'C': 169.15690047953544, 'kernel': 'sigmoid', 'gamma': 0.008783749726916715},
     {'C': 12.835771485641382, 'kernel': 'sigmoid', 'gamma': 1.0071596631848516},
     {'C': 30.551114355393477, 'kernel': 'poly', 'gamma': 0.0011663605324968745},
     {'C': 896.6682272351945, 'kernel': 'sigmoid', 'gamma': 0.02971616148132878},
     {'C': 284.9975261801749, 'kernel': 'poly', 'gamma': 3.882373476712721e-05},
     {'C': 0.12580143403720118, 'kernel': 'poly', 'gamma': 3.882373476712721e-05},
     {'C': 4.421473749743905, 'kernel': 'poly', 'gamma': 1091.8068066124756},
     {'C': 0.12580143403720118, 'kernel': 'poly', 'gamma': 0.0005766299499622274},
     {'C': 896.6682272351945, 'kernel': 'poly', 'gamma': 3.882373476712721e-05}]



```python
# The extra information is stored in each individual.
[i.extra for i in population]
```




    [[0.9277777777777778],
     [0.06944444444444445],
     [0.06944444444444445],
     [0.9944444444444445],
     [0.06944444444444445],
     [1.0],
     [0.06944444444444445],
     [0.9944444444444445],
     [1.0],
     [1.0]]



Print Hall of Fame with extra info:

```python
[(i,i.extra) for i in hof.items]
```




    [({'C': 0.12580143403720118, 'kernel': 'poly', 'gamma': 0.0005766299499622274},
      [1.0]),
     ({'C': 896.6682272351945, 'kernel': 'poly', 'gamma': 3.882373476712721e-05},
      [1.0])]



Use SVC with probability support, which is much slower than the default SVC.

```python
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred.argmax(-1))

def evaluate_proba(params):
    return svc_digits_proba.val_loss_with_params(callbacks=(accuracy,), **params)

toolbox.register('evaluate', evaluate_proba)
hof.clear()
pmap.close()
pmap = ConcurrentMap(10)
toolbox.register('map', pmap.map)
def run_proba():
    return eaSimpleWithExtraLog(toolbox.population(10),
                                toolbox,
                                cxpb=0.6,
                                mutpb=0.6,
                                ngen=16,
                                halloffame=hof,
                                elitism=True,
                                stats=stats)
```

```python
%time population, logbook = run_proba()
```

    gen	nevals	avg         	std         	min         	max         
    0  	10    	[1.13739406]	[1.05320105]	[0.08716575]	[2.30945503]
    1  	6     	[0.31907472]	[0.65890972]	[0.08716575]	[2.29541009]
    2  	6     	[0.75435098]	[1.01507315]	[0.08716575]	[2.30968647]
    3  	6     	[0.52748718]	[0.86342645]	[0.08669759]	[2.29641406]
    4  	8     	[0.5315929] 	[0.88268664]	[0.08669759]	[2.29748447]
    5  	7     	[1.0139067] 	[1.01525905]	[0.08669759]	[2.30946469]
    6  	7     	[0.41403803]	[0.6993707] 	[0.08669759]	[2.30956588]
    7  	7     	[0.75477561]	[1.01133589]	[0.08669759]	[2.30956588]
    8  	6     	[0.63293026]	[0.85851073]	[0.08423906]	[2.30962038]
    9  	7     	[1.19521929]	[1.10602421]	[0.08423906]	[2.31397709]
    10 	8     	[1.36054036]	[1.05278286]	[0.08423906]	[2.31156383]
    11 	6     	[0.8192004] 	[0.89424649]	[0.08423906]	[2.30442492]
    12 	6     	[1.00826583]	[1.06227027]	[0.08423906]	[2.30950605]
    13 	8     	[0.76115428]	[1.01018047]	[0.08369347]	[2.30943281]
    14 	6     	[0.31281143]	[0.66295653]	[0.08369347]	[2.30156706]
    15 	5     	[0.09080058]	[0.00761646]	[0.08369347]	[0.10120453]
    16 	6     	[1.32092735]	[1.04753049]	[0.08423906]	[2.31144868]
    CPU times: user 150 ms, sys: 56.3 ms, total: 206 ms
    Wall time: 36 s


```python
[(i,i.extra) for i in hof.items]
```




    [({'C': 0.7859445686841542, 'kernel': 'rbf', 'gamma': 0.0010295029455246714},
      [0.9888888888888889]),
     ({'C': 372.6869250600894, 'kernel': 'rbf', 'gamma': 0.0010295029455246714},
      [0.9861111111111112])]



```python
hof.clear()
pmap.close()
pmap = ConcurrentMap(10)
toolbox.register('evaluate', evaluate)
toolbox.register('map', pmap.map)
def run_mu_plus_lambda():
    return eaMuPlusLambdaWithExtraLog(toolbox.population(10),
                                toolbox,
                                mu=6,
                                lambda_=10,
                                cxpb=0.3,
                                mutpb=0.6,
                                ngen=16,
                                halloffame=hof,
                                stats=stats)
```

```python
%time population, logbook = run_mu_plus_lambda()
```

    gen	nevals	avg          	std         	min          	max          
    0  	10    	[-0.43861111]	[0.45214074]	[-0.99444444]	[-0.06944444]
    1  	8     	[-0.99490741]	[0.00103522]	[-0.99722222]	[-0.99444444]
    2  	10    	[-0.99444444]	[0.]        	[-0.99444444]	[-0.99444444]
    3  	7     	[-0.99490741]	[0.00103522]	[-0.99722222]	[-0.99444444]
    4  	8     	[-0.9962963] 	[0.00130946]	[-0.99722222]	[-0.99444444]
    5  	10    	[-0.9962963] 	[0.00130946]	[-0.99722222]	[-0.99444444]
    6  	9     	[-0.97314815]	[0.05011988]	[-0.99722222]	[-0.86111111]
    7  	9     	[-0.99490741]	[0.00103522]	[-0.99722222]	[-0.99444444]
    8  	10    	[-0.99490741]	[0.00103522]	[-0.99722222]	[-0.99444444]
    9  	10    	[-0.99490741]	[0.00103522]	[-0.99722222]	[-0.99444444]
    10 	8     	[-0.99583333]	[0.00138889]	[-0.99722222]	[-0.99444444]
    11 	8     	[-0.84212963]	[0.34555681]	[-0.99722222]	[-0.06944444]
    12 	8     	[-0.99722222]	[1.11022302e-16]	[-0.99722222]	[-0.99722222]
    13 	9     	[-0.99722222]	[1.11022302e-16]	[-0.99722222]	[-0.99722222]
    14 	7     	[-0.99722222]	[1.11022302e-16]	[-0.99722222]	[-0.99722222]
    15 	9     	[-0.99722222]	[1.11022302e-16]	[-0.99722222]	[-0.99722222]
    16 	7     	[-0.99814815]	[0.00130946]    	[-1.]        	[-0.99722222]
    CPU times: user 173 ms, sys: 44.9 ms, total: 218 ms
    Wall time: 8.87 s


```python
[(i,i.extra) for i in hof.items]
```




    [({'C': 1.300279542249852, 'kernel': 'rbf', 'gamma': 0.0003904230219527077},
      [1.0]),
     ({'C': 22.933308169323965, 'kernel': 'rbf', 'gamma': 0.0003904230219527077},
      [0.9972222222222222])]



```python
hof.clear()
pmap.close()
pmap = ConcurrentMap(10)
toolbox.register('evaluate', evaluate)
toolbox.register('map', pmap.map)
def run_mu_comma_lambda():
    return eaMuCommaLambdaWithExtraLog(toolbox.population(10),
                                toolbox,
                                mu=6,
                                lambda_=16,
                                cxpb=0.3,
                                mutpb=0.6,
                                ngen=16,
                                halloffame=hof,
                                stats=stats)
```

```python
%time population, logbook = run_mu_plus_lambda()
```

    gen	nevals	avg          	std     	min          	max          
    0  	10    	[-0.53194444]	[0.4625]	[-0.99444444]	[-0.06944444]
    1  	8     	[-0.84027778]	[0.34472715]	[-0.99444444]	[-0.06944444]
    2  	10    	[-0.82962963]	[0.34076464]	[-0.99444444]	[-0.06944444]
    3  	9     	[-0.99444444]	[0.]        	[-0.99444444]	[-0.99444444]
    4  	10    	[-0.9962963] 	[0.00261891]	[-1.]        	[-0.99444444]
    5  	9     	[-0.99537037]	[0.00207043]	[-1.]        	[-0.99444444]
    6  	10    	[-0.9962963] 	[0.00261891]	[-1.]        	[-0.99444444]
    7  	9     	[-0.99814815]	[0.00261891]	[-1.]        	[-0.99444444]
    8  	10    	[-1.]        	[0.]        	[-1.]        	[-1.]        
    9  	9     	[-1.]        	[0.]        	[-1.]        	[-1.]        
    10 	6     	[-1.]        	[0.]        	[-1.]        	[-1.]        
    11 	9     	[-0.99907407]	[0.00207043]	[-1.]        	[-0.99444444]
    12 	9     	[-1.]        	[0.]        	[-1.]        	[-1.]        
    13 	10    	[-1.]        	[0.]        	[-1.]        	[-1.]        
    14 	10    	[-0.99907407]	[0.00207043]	[-1.]        	[-0.99444444]
    15 	9     	[-0.99907407]	[0.00207043]	[-1.]        	[-0.99444444]
    16 	9     	[-1.]        	[0.]        	[-1.]        	[-1.]        
    CPU times: user 179 ms, sys: 38.4 ms, total: 218 ms
    Wall time: 7.9 s


```python
[(i,i.extra) for i in hof.items]
```




    [({'C': 73.66617357942974, 'kernel': 'poly', 'gamma': 9.64738772970552e-05},
      [1.0]),
     ({'C': 0.4031809078286158, 'kernel': 'poly', 'gamma': 0.0003899126396372798},
      [1.0])]



### Optimize parameters based on CV result

```python
from eptune.sample_cases import cv_svc_digits
from sklearn.model_selection import StratifiedKFold


def cv_evaluate(params):
    return cv_svc_digits.cv_loss_with_params(callbacks=(accuracy_score, ),
                                             cv=StratifiedKFold(n_splits=3),
                                             **params)


hof.clear()
pmap.close()
pmap = ConcurrentMap(10)
toolbox.register('evaluate', cv_evaluate)
toolbox.register('map', pmap.map)


def runcv():
    return eaSimpleWithExtraLog(toolbox.population(10),
                                toolbox,
                                cxpb=0.6,
                                mutpb=0.6,
                                ngen=16,
                                halloffame=hof,
                                elitism=True,
                                stats=stats)
```

```python
%time population, logbook = runcv()
```

    gen	nevals	avg          	std         	min          	max          
    0  	10    	[-0.10300501]	[0.00421203]	[-0.11519199]	[-0.10072343]
    1  	8     	[-0.44991653]	[0.42371549]	[-0.96883695]	[-0.10072343]
    2  	6     	[-0.46850306]	[0.41263473]	[-0.96883695]	[-0.10072343]
    3  	7     	[-0.81341124]	[0.30921735]	[-0.96883695]	[-0.10072343]
    4  	7     	[-0.70628826]	[0.39648358]	[-0.96883695]	[-0.10072343]
    5  	7     	[-0.88074569]	[0.25893916]	[-0.96883695]	[-0.10406233]
    6  	7     	[-0.72765721]	[0.36969243]	[-0.96883695]	[-0.10072343]
    7  	6     	[-0.88230384]	[0.25959933]	[-0.96883695]	[-0.10350584]
    8  	5     	[-0.79632721]	[0.34516512]	[-0.96939343]	[-0.10127991]
    9  	8     	[-0.88224819]	[0.25995182]	[-0.96939343]	[-0.10239288]
    10 	8     	[-0.96755704]	[0.00421644]	[-0.96939343]	[-0.95492487]
    11 	7     	[-0.85731775]	[0.26229403]	[-0.96939343]	[-0.10350584]
    12 	6     	[-0.79549249]	[0.34724552]	[-0.96939343]	[-0.10072343]
    13 	7     	[-0.88786867]	[0.24383245]	[-0.96939343]	[-0.15637173]
    14 	7     	[-0.86944908]	[0.25897708]	[-0.9705064] 	[-0.10127991]
    15 	8     	[-0.52754591]	[0.42718517]	[-0.9705064] 	[-0.10072343]
    16 	8     	[-0.78224819]	[0.3425227] 	[-0.9705064] 	[-0.10127991]
    CPU times: user 147 ms, sys: 74.1 ms, total: 222 ms
    Wall time: 39.1 s


```python
hof.clear()
pmap.close()
pmap = ConcurrentMap(10)
toolbox.register('evaluate', cv_evaluate)
toolbox.register('map', pmap.map)
def runcv_mu_plus_lambda():
    return eaMuPlusLambdaWithExtraLog(toolbox.population(10),
                                toolbox,
                                mu=6,
                                lambda_=10,
                                cxpb=0.3,
                                mutpb=0.6,
                                ngen=16,
                                halloffame=hof,
                                stats=stats)
```

```python
%time population, logbook = runcv_mu_plus_lambda()
```

    gen	nevals	avg          	std         	min          	max          
    0  	10    	[-0.36816917]	[0.38938642]	[-0.96883695]	[-0.10072343]
    1  	10    	[-0.82674828]	[0.30909141]	[-0.96883695]	[-0.13578186]
    2  	10    	[-0.82999444]	[0.3104613] 	[-0.96883695]	[-0.13578186]
    3  	8     	[-0.83361157]	[0.30237313]	[-0.96883695]	[-0.1574847] 
    4  	9     	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    5  	10    	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    6  	9     	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    7  	8     	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    8  	9     	[-0.81867928]	[0.32106232]    	[-0.96883695]	[-0.10127991]
    9  	10    	[-0.96327212]	[0.01244334]    	[-0.96883695]	[-0.93544797]
    10 	9     	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    11 	8     	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    12 	10    	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    13 	10    	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    14 	9     	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    15 	10    	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    16 	10    	[-0.68094973]	[0.40713401]    	[-0.96883695]	[-0.10517529]
    CPU times: user 179 ms, sys: 77.5 ms, total: 256 ms
    Wall time: 40.2 s


```python
hof.clear()
pmap.close()
pmap = ConcurrentMap(10)
toolbox.register('evaluate', cv_evaluate)
toolbox.register('map', pmap.map)
def runcv_mu_comma_lambda():
    return eaMuCommaLambdaWithExtraLog(toolbox.population(10),
                                toolbox,
                                mu=6,
                                lambda_=16,
                                cxpb=0.3,
                                mutpb=0.6,
                                ngen=16,
                                halloffame=hof,
                                stats=stats)
```

```python
%time population, logbook = runcv_mu_comma_lambda()
```

    gen	nevals	avg          	std         	min          	max          
    0  	10    	[-0.36688926]	[0.39504808]	[-0.97273233]	[-0.10072343]
    1  	13    	[-0.67677611]	[0.39581716]	[-0.96883695]	[-0.10294936]
    2  	15    	[-0.79150436]	[0.31545629]	[-0.96994992]	[-0.10573178]
    3  	16    	[-0.81867928]	[0.29654357]	[-0.96883695]	[-0.15637173]
    4  	13    	[-0.9689297] 	[0.00148685]	[-0.9705064] 	[-0.96605454]
    5  	15    	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    6  	14    	[-0.96920794]	[0.00082956]    	[-0.97106288]	[-0.96883695]
    7  	15    	[-0.9689297] 	[0.00148685]    	[-0.9705064] 	[-0.96605454]
    8  	16    	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    9  	16    	[-0.9655908] 	[0.00725861]    	[-0.96883695]	[-0.94936004]
    10 	13    	[-0.9705064] 	[0.00236096]    	[-0.9738453] 	[-0.96883695]
    11 	14    	[-0.83008718]	[0.31025391]    	[-0.96883695]	[-0.13633834]
    12 	14    	[-0.82415136]	[0.32352681]    	[-0.96883695]	[-0.10072343]
    13 	14    	[-0.82415136]	[0.32352681]    	[-0.96883695]	[-0.10072343]
    14 	14    	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    15 	15    	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    16 	14    	[-0.96883695]	[1.11022302e-16]	[-0.96883695]	[-0.96883695]
    CPU times: user 228 ms, sys: 64 ms, total: 292 ms
    Wall time: 44 s


```python
[(i,i.extra) for i in hof.items]
```




    [({'C': 13.609159696697825, 'kernel': 'rbf', 'gamma': 0.0006613868491920236},
      [0.9738452977184195]),
     ({'C': 39.37419618476607, 'kernel': 'rbf', 'gamma': 0.00030793120799049256},
      [0.9732888146911519])]


