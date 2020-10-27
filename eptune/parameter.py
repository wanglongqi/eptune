# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_parameter.ipynb (unless otherwise specified).

__all__ = ['Parameter', 'OrderedParameter', 'IntegerParameter', 'InListNumericParameter', 'FloatParameter',
           'LogFloatParameter', 'CategoricalParameter', 'CallableParameter']

# Cell
from enum import Enum
import abc
from random import random, randint, normalvariate as randn
from functools import partial
from math import log, log10, log2
from .utils import logx

class Parameter:
    "Base class for parameter."
    def __init__(self, values, name=None, default_value=None):
        if name is None:
            self.name = str(self.__class__.__name__)
        else:
            self.name = str(name)
        self.set_values(values)
        if default_value is None:
            self.default_value = self.get_rand_value()
        else:
            self.default_value = default_value

    def __repr__(self):
        return f'{self.__class__.__name__}(values={self.values}, name="{self.name}", default_value={self.default_value})'

    @abc.abstractmethod
    def set_values(self, values):
        raise NotImplementedError('Not implemented')

    @abc.abstractmethod
    def get_rand_value(self, *args, **kwargs):
        raise NotImplementedError('Not implemented')

    def __next__(self):
        return self.get_rand_value()

    @abc.abstractmethod
    def is_valid(self, value, fix=False):
        raise NotImplementedError('Not implemented')

# Cell
class OrderedParameter(Parameter):
    "Base class for parameter that is ordered, such as numerical parameters."
    @abc.abstractmethod
    def get_value(self, ratio=0):
        raise NotImplementedError('Not implemented')

# Cell
class IntegerParameter(OrderedParameter):
    "Integer pramameter with a range."
    def set_values(self, values):
        if len(values) != 2:
            raise ValueError(
                f"values should have and only have two elements as lower and upper bound of the value"
            )
        self.values = sorted(values)

    def get_rand_value(self):
        return randint(*self.values)

    def get_value(self, ratio=0):
        if ratio > 1: ratio = 1
        if ratio < 0: ratio = 0
        return self.values[0] + round(
            (self.values[1] - self.values[0]) * ratio)

    def is_valid(self, value, fix=False):
        if value>=self.values[0]:
            if value <= self.values[1]:
                if fix:
                    return value
                else:
                    return True
            else:
                if fix:
                    return self.values[1]
                else:
                    return False
        else:
            if fix:
                return self.values[0]
            else:
                return False

# Cell
import bisect
class InListNumericParameter(OrderedParameter):
    """Numerical pramameter with a known set of values, and the values are ordered.
    Otherwise, it becomes a categorical parameter."""
    def set_values(self, values):
        self.values = sorted(values)
        self._len = len(self.values)

    def get_rand_value(self):
        return self.values[randint(1, self._len) - 1]

    def get_value(self, ratio=0):
        if ratio > 1: ratio = 1
        if ratio < 0: ratio = 0
        return self.values[round(self._len * ratio)]

    def is_valid(self, value, fix=False):
        if value in self.values:
            if fix:
                return value
            else:
                return True
        else:
            if fix:
                idx = bisect.bisect(self.values, value)
                idx = idx-1 if idx==self._len else idx
                return self.values[idx]
            else:
                return False

# Cell
class FloatParameter(OrderedParameter):
    "Floating number parameter with a range."

    def set_values(self, values):
        if len(values) != 2:
            raise ValueError(
                f"values should have and only have two elements as lower and upper bound of the value"
            )
        self.values = sorted(values)
        self._range = self.values[1] - self.values[0]
        self._left = self.values[0]

    def get_rand_value(self, a=None, b=None):
        if a is None or b is None:
            return random() * self._range + self._left
        else:
            return random() * abs(a - b) + a if a < b else b

    def get_value(self, ratio=0):
        if ratio > 1: ratio = 1
        if ratio < 0: ratio = 0
        return self._range * ratio + self._left

    def is_valid(self, value, fix=False):
        if value >= self.values[0]:
            if value <= self.values[1]:
                if fix:
                    return value
                else:
                    return True
            else:
                if fix:
                    return self.values[1]
                else:
                    return False
        else:
            if fix:
                return self.values[0]
            else:
                return False

# Cell
class LogFloatParameter(OrderedParameter):
    """Floating number parameter with a range, but the sampling is in a logrithmic scale.
    So lower paramter range is sampled more frequentyly than higher range.

    - Note: the parameter range must be positive, as `log` of negative number is not a real number.
    """
    def __init__(self, values, name=None, default_value=None):
        super().__init__(values, name, default_value=default_value)

    def set_values(self, values):
        if len(values) != 2:
            raise ValueError(
                f"values should have and only have two elements as lower and upper bound of the value"
            )
        self.values = sorted(values)
        self._left = log10(self.values[0])
        self._right = log10(self.values[1])
        self._range = self._right - self._left

    def get_rand_value(self, a=None, b=None):
        if a is None or b is None:
            return 10**(random() * self._range + self._left)
        else:
            a = log10(a)
            b = log10(b)
            return 10**(random() * abs(a - b) + a if a < b else b)

    def get_value(self, ratio=0):
        if ratio > 1: ratio = 1
        if ratio < 0: ratio = 0
        a = self._left
        b = self._right
        return 10**(ratio * abs(a - b) + a if a < b else b)

    def is_valid(self, value, fix=False):
        if value >= self.values[0]:
            if value <= self.values[1]:
                if fix:
                    return value
                else:
                    return True
            else:
                if fix:
                    return self.values[1]
                else:
                    return False
        else:
            if fix:
                return self.values[0]
            else:
                return False

# Cell
class CategoricalParameter(Parameter):
    "Categorical parameter"
    def set_values(self, values):
        self.values = list(set(values))
        self._len = len(self.values)

    def get_rand_value(self):
        return self.values[randint(1, self._len) - 1]

    def is_valid(self, value, fix=False):
        if value in self.values:
            if fix:
                return value
            else:
                return True
        else:
            if fix:
                return self.values[randint(1, self._len) - 1]
            else:
                return False

# Cell
class CallableParameter(Parameter):
    """The values of the parameter is a callable. When execute the values attribute,
    the callable will return the possible value of the parameter."""
    def set_values(self, values):
        if callable(values):
            self.values = values
        else:
            raise ValueError('values need to be a callable object.')

    def get_rand_value(self, *args, **kwargs):
        return self.values(*args, **kwargs)