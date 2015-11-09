__author__ = 'quynhdo'
#multi-layer perceptron
from liir.nlp.ml.nn.base.Layer  import Layer
from liir.nlp.ml.nn.base.NNNet  import NNNet
from liir.nlp.ml.nn.base.Functions import TanhOutputFunction, SoftmaxOutputFunction, SquaredErrorCostFunction
from liir.nlp.ml.nn.base.Connection import Connection
import theano
import numpy
class MLP(NNNet):


    def __init__(self, layers, size_output_layer=100,  activate_function=TanhOutputFunction, cost_function=SquaredErrorCostFunction, input=None, input_type="matrix", output=None,  id=""):

        NNNet.__init__(self, layers=layers, cost_function=cost_function, input=input, output=output, input_type=input_type, output_type="vector")
        for i in range(len(layers)):
            self.layers[i].id = id + str(i)

        if layers[len(layers)-1].ltype != Layer.Layer_Type_Output:
            output_layer = Layer(numNodes=size_output_layer, ltype = Layer.Layer_Type_Output, id=id+str(len(layers)))
            self.layers.append(output_layer)

        rng = numpy.random.RandomState(123)
        for i in range(len(self.layers)-1):
            c = None
            if i <len(self.layers)-2:
                c = self.createConnection(self.layers[i],self.layers[i+1], of = activate_function)
            else:
                c = self.createConnection(self.layers[i],self.layers[i+1], of = SoftmaxOutputFunction, otype=Connection.Output_Type_SoftMax)

            self.connections.append(c)
            self.params=self.params+c.params


        self.connect(self.x)



