__author__ = 'quynhdo'
import theano as th
import theano.tensor as T
import timeit
import numpy as np

from liir.nlp.ml.nn.base.Functions import DotActivateFunction
from liir.nlp.ml.nn.base.Functions import SigmoidOutputFunction
from liir.nlp.ml.nn.base.Connection import  Connection
class NNNet:

    def __init__(self, *layers, cost_function, **connection_config):
        self.layers=[]
        self.connections=[]
        self.layers=layers
        self.params=[]
        if len(connection_config)!=0:
            for l1,l2 in connection_config.keys():
                af,of=connection_config[(l1,l2)]
                c = self.createConnection(layers[l1],layers[l2], af, of)
                self.connections.append(c)
                self.params=self.params+c.params
        else:
            for i in range(len(layers)-1):
                c = self.createConnection(layers[i],layers[i+1])
                self.connections.append(c)
                self.params=self.params+c.params

        self.cost_function= cost_function



    def createConnection (self, l1, l2, af= DotActivateFunction, of = SigmoidOutputFunction):
        return Connection(scr=l1, dst=l2, activate_func=af, output_func=of, use_bias=l1.useBias , id="c"+l1.id)


    def get_connection(self, l1, l2):
        for conn in self.connections:
            if conn.scr == l1 and conn.dst == l2:
                return conn
        return None

    def connect(self, x):
        assert len(self.layers) >=2
        self.layers[0].process_input(x)
        for conn in self.connections:
            conn.connect()




    def get_cost_updates(self, x, y, learning_rate):
        self.connect(x)
        cost = self.cost_function(self.layers[len(self.layers)-1].output , y)

        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

    def fit(self, train_data, train_data_label, batch_size, training_epochs, learning_rate):
        index = T.lscalar()
        x = T.matrix('x')
        if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_SoftMax):
            y = T.ivector('y')
        if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_Binary):
            y = T.iscalar('y')
        if (self.connections[len(self.connections)-1].otype == Connection.Output_Type_Real):
            y = T.matrix('y')

        cost,updates=self.get_cost_updates(x,y, learning_rate)
        train_da = th.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_data[index * batch_size: (index + 1) * batch_size],
            y: train_data_label[index * batch_size: (index + 1) * batch_size]
            }
        )

        n_train_batches = (int) (train_data_label.get_value(borrow=True).shape[0] / batch_size)
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



