__author__ = 'quynhdo'
import theano as th
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams


#### activate functions

def DotActivateFunction( x, W, b):
        if b !=None:
            return T.dot(x, W) + b
        else:
            return T.dot(x,W)

def NoneActivateFunction(self, x, W, b):
        return x



#### output functions
SigmoidOutputFunction = T.nnet.sigmoid
SoftmaxOutputFunction = T.nnet.softmax
def NoneOutputFunction (x):
    return x



#### cost functions
def NegativeLogLikelihoodCostFunction(o, y):
        return -T.mean(T.log(o)[T.arange(y.shape[0]), y])

def SquaredErrorCostFunction(o,y):
        return -T.mean((o-y) ** 2)

def CrossEntroyCostFunction(o,y):
    L = - T.sum(y * T.log(o) + (1 - y) * T.log(1 - o), axis=1)

    cost = T.mean(L)
    return cost



#### functions to process input for the input layer
rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

def NoneProcessFunction(x, *args): return x
def DenoisingProcessFunction(x, corruption_level):  return theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level, dtype=th.config.floatX) * x