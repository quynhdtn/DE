from liir.nlp.ml.nn.base.Model import Model
from liir.nlp.utils.Enum import Enum

__author__ = 'quynhdo'
import numpy as np

import theano as th
from liir.nlp.ml.nn.base.Functions import NoneProcessFunction
# this class define a Layer in neural network
class Layer:

    Layer_Type_Input="input"
    Layer_Type_Hidden="hidden"
    Layer_Type_Output="output"


    def __init__(self, numNodes, ltype, useBias=True, id="", input_process_func=NoneProcessFunction):
        self.ltype = ltype
        self.output = np.zeros(numNodes)
        self.useBias = useBias
        self.size = numNodes
        self.id = id
        self.input_process_func = input_process_func

        if ltype == Layer.Layer_Type_Output:
            self.label = None


    def process_input(self, x, *kargs):
        self.output = self.input_process_func(x, kargs)






























