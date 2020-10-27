# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_mutation.ipynb (unless otherwise specified).

__all__ = ['mutDictRand', 'mutRand']

# Cell
import random

def mutDictRand(individual, params=None, indpb=0.6):
    "Mutation function (for dict base class) that changes the value of an individual at the probability of `indpb`."
    if params==None:
        params = individual.params
    for p in params:
        if p.name in individual:
            if random.random() < indpb:
                individual[p.name] = next(p)
    return individual,

def mutRand(individual, params=None, indpb=0.6):
    "Mutation function (for list base class) that changes the value of an individual at the probability of `indpb`."
    if params==None:
        params = individual.params
    for idx, p in enumerate(params):
        if random.random() < indpb:
            individual[idx] = next(p)
    return individual,