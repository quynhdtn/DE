__author__ = 'quynhdo'
#multi-layer perceptron
from liir.nlp.ml.nn.base.Layer  import Layer
from liir.nlp.ml.nn.base.NNNet  import NNNet
from liir.nlp.ml.nn.base.Functions import TanhOutputFunction, SoftmaxOutputFunction, SquaredErrorCostFunction
from liir.nlp.ml.nn.base.Connection import Connection
import theano.tensor as T
import theano as th
import numpy as np
import timeit

class MLP(NNNet):


    def __init__(self, layers, size_output_layer=100,  activate_function=TanhOutputFunction, cost_function=SquaredErrorCostFunction, input=None, extend_input=None, input_type="matrix", output=None,  id=""):

        NNNet.__init__(self, layers=layers, cost_function=cost_function, input=input, output=output, input_type=input_type, output_type="vector")
        for i in range(len(layers)):
            self.layers[i].id = id + str(i)

        if layers[len(layers)-1].ltype != Layer.Layer_Type_Output:
            output_layer = Layer(numNodes=size_output_layer, ltype = Layer.Layer_Type_Output, id=id+str(len(layers)))
            self.layers.append(output_layer)

        rng = np.random.RandomState(123)
        for i in range(len(self.layers)-1):
            c = None
            if i <len(self.layers)-2:
                c = self.createConnection(self.layers[i],self.layers[i+1], of = activate_function)
            else:
                c = self.createConnection(self.layers[i],self.layers[i+1], of = SoftmaxOutputFunction, otype=Connection.Output_Type_SoftMax)

            self.connections.append(c)
            self.params=self.params+c.params
        if extend_input is None:
            if input_type == "matrix":
                self.x_extend = T.matrix(name='extend_input')

            if input_type == "vector":
                self.x = T.vector(name='extend_input')
        else:
            self.x_extend = extend_input

        self.layers[len(layers)-2].extend=self.x_extend

        self.connect(self.x)

    def fit(self, train_data, train_data_extend, train_data_label, batch_size, training_epochs, learning_rate):

        index = T.lscalar()
      #  x = T.matrix('x')
     #   if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_SoftMax):
     #       y = T.ivector('y')

      #  if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_Binary):
      #      y = T.iscalar('y')
      #  if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_Real):
       # y = T.matrix('y')

        cost,updates=self.get_cost_updates(learning_rate)
        train_da = th.function(
        [index],
        cost,
        updates=updates,
        givens={
            self.x: train_data[index * batch_size: (index + 1) * batch_size],
            self.y: train_data_label[index * batch_size: (index + 1) * batch_size],
            self.x_extend: train_data_extend[index * batch_size: (index + 1) * batch_size]
            }
        )
        n_train_batches = (int) (train_data.get_value(borrow=True).shape[0] / batch_size)

     #   n_train_batches =2
        start_time = timeit.default_timer()
        for epoch in range(training_epochs):
        # go through trainng set
            c = []
            for batch_index in range(int(n_train_batches)):
                c.append(train_da(batch_index))

            print ('Training epoch %d, cost ' % epoch, np.mean(c))

        end_time = timeit.default_timer()

        training_time = (end_time - start_time)
        print('Training time: %.2fm' % ((training_time) / 60.))

    def predict(self, x, x_extend):
        t=x
        for i in range(len(self.connections)-1):
            t = self.connections[i].getOutputValue(t)

        return self.connections[len(self.connections)-1].getOutputValueExtended(t, x_extend)

class MLPExtended(NNNet):


    def __init__(self, layers, size_output_layer=100,  activate_function=TanhOutputFunction, cost_function=SquaredErrorCostFunction, input=None, extend_input=None,  input_type="matrix", output=None,  id=""):

        NNNet.__init__(self, layers=layers, cost_function=cost_function, input=input, output=output, input_type=input_type, output_type="vector")
        for i in range(len(layers)):
            self.layers[i].id = id + str(i)

        if layers[len(layers)-1].ltype != Layer.Layer_Type_Output:
            output_layer = Layer(numNodes=size_output_layer, ltype = Layer.Layer_Type_Output, id=id+str(len(layers)))
            self.layers.append(output_layer)



        rng = np.random.RandomState(123)
        for i in range(len(self.layers)-1):
            c = None
            if i <len(self.layers)-2:
                c = self.createConnection(self.layers[i],self.layers[i+1], of = activate_function)
            else:
                c = self.createConnection(self.layers[i],self.layers[i+1], of = SoftmaxOutputFunction, otype=Connection.Output_Type_SoftMax)

            self.connections.append(c)
            self.params=self.params+c.params


        if extend_input is None:
            if input_type == "matrix":
                self.x_extend = T.matrix(name='extend_input')

            if input_type == "vector":
                self.x = T.vector(name='extend_input')
        else:
            self.x_extend = extend_input



        self.layers[len(layers)-2].extend=self.x_extend
        self.connect(self.x)



    def fit(self, train_data, train_data_extend, train_data_label, batch_size, training_epochs, learning_rate):

        index = T.lscalar()
      #  x = T.matrix('x')
     #   if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_SoftMax):
     #       y = T.ivector('y')

      #  if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_Binary):
      #      y = T.iscalar('y')
      #  if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_Real):
       # y = T.matrix('y')

        cost,updates=self.get_cost_updates(learning_rate)
        train_da = th.function(
        [index],
        cost,
        updates=updates,
        givens={
            self.x: train_data[index * batch_size: (index + 1) * batch_size],
            self.y: train_data_label[index * batch_size: (index + 1) * batch_size],
            self.extend: train_data_extend[index * batch_size: (index + 1) * batch_size]
            }
        )
        n_train_batches = (int) (train_data.get_value(borrow=True).shape[0] / batch_size)

     #   n_train_batches =2
        start_time = timeit.default_timer()
        for epoch in range(training_epochs):
        # go through trainng set
            c = []
            for batch_index in range(int(n_train_batches)):
                c.append(train_da(batch_index))

            print ('Training epoch %d, cost ' % epoch, np.mean(c))

        end_time = timeit.default_timer()

        training_time = (end_time - start_time)
        print('Training time: %.2fm' % ((training_time) / 60.))

    def predict(self, x, x_extend):
        t=x
        for i in range(len(self.connections)-1):
            t = self.connections[i].getOutputValue(t)

        return self.connections[len(self.connections)-1].getOutputValue(t, x_extend)