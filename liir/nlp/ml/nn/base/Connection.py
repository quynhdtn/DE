__author__ = 'quynhdo'

import theano as th
import numpy as np
import theano.tensor as T
from liir.nlp.ml.nn.base.Layer import Layer
from theano.tensor.shared_randomstreams import RandomStreams
# implement the connection between two layers
class Connection:


    Output_Type_Real="real"
    Output_Type_Binary="binary"
    Output_Type_SoftMax="softmax"



    def __init__(self, scr, dst, activate_func, output_func, use_bias=True, id="", initial_w = None, otype="real" ):
        self.scr = scr  # source layer
        self.dst = dst  # destination layer
        self.activate_func = activate_func # transfer function
        self.output_func = output_func  # activate function
        self.otype=otype
        self.W=th.shared(value=np.zeros((scr.size, dst.size), dtype=th.config.floatX), name="W" + id, borrow=True)
        self.rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))
        self.W = th.shared(value=np.asarray(
                self.rng.uniform(
                    low=-4 * np.sqrt(6. / (scr.size + dst.size)),
                    high=4 * np.sqrt(6. / (scr.size + dst.size)),
                    size=(scr.size, dst.size)
                ),
                dtype=th.config.floatX
            ),name='W', borrow=True)

        if initial_w != None:
            self.W = th.shared(value=initial_w, dtype=th.config.floatX, name="W" + id, borrow=True)

        self.params=[self.W]
        if use_bias:
            self.b = th.shared(value=np.zeros(dst.size), name="b" + id, borrow=True)
            self.params.append(self.b)
        else:
            self.b = None

    # when connect is called, the output of dst layer is calculated
    def connect(self):
        self.dst.output = self.output_func(self.activate_func(self.scr.output, self.W, self.b))
        if self.dst.ltype == Layer.Layer_Type_Output:
            self.computeOutput(self.dst.output)


    def computeOutput(self,y_pred):

        if self.otype == Connection.Output_Type_Binary:
            self.dst.output = T.round(y_pred)

        if self.otype == Connection.Output_Type_SoftMax:
            self.dst.output = T.argmax(y_pred, axis=1)






