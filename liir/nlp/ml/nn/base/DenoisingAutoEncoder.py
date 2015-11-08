__author__ = 'quynhdo'
import theano as th
from liir.nlp.ml.nn.base.AutoEncoder import AutoEncoder
from liir.nlp.ml.classifiers.linear.logistic import load_data
try:
    import PIL.Image as Image
except ImportError:
    import Image
from liir.nlp.ml.classifiers.linear.utils import tile_raster_images
from liir.nlp.ml.nn.base.Layer  import Layer,DenoisingLayer
from liir.nlp.ml.nn.base.NNNet  import NNNet
from liir.nlp.ml.nn.base.Functions import CrossEntroyCostFunction
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import numpy as np

class DenoisingAutoEncoder(NNNet):
    def __init__(self,nIn=700, nHidden=500, corruption_level=0.1,id=""):

        self.corruption_level = corruption_level

        self.id=id
        self.nIn=nIn
        self.nHidden=nHidden

        ilayer = DenoisingLayer(numNodes=nIn, ltype = Layer.Layer_Type_Input, id=id+"0", corruption_level=corruption_level)
        hlayer = Layer(numNodes=nHidden, ltype = Layer.Layer_Type_Hidden, id=id+"1")
        olayer = Layer(numNodes=nIn, ltype = Layer.Layer_Type_Output, id=id+"2")


        # declare nnnet
        NNNet.__init__(self,ilayer, hlayer, olayer, cost_function=CrossEntroyCostFunction)

        # change parameter constraint
        conn1 = self.connections[0]
        conn2 = self.connections[1]

        self.params.remove(conn2.W)
        conn2.W = conn1.W.T


    def get_cost_updates(self, x, y, learning_rate):


        self.connect(x)
        cost = self.cost_function(self.layers[len(self.layers)-1].output , y)

        if self.knowledge != None:
            h = self.layers[len(self.layers)-2].output
            for p in self.knowledge['pair']:
                x1= h[p[0]]
                x2= h[p[1]]


                cost= cost - T.mean ((x1-x2)**2)

        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)





    def fit(self, train_data, batch_size, training_epochs, learning_rate, knowledge=None):
        self.knowledge=knowledge
        NNNet.fit(self,train_data, train_data, batch_size, training_epochs, learning_rate)
        image = Image.fromarray(
            tile_raster_images(X=self.connections[0].W.get_value(borrow=True).T,
                               img_shape=(np.sqrt(self.nIn), np.sqrt(self.nIn)), tile_shape=(10, 10),
                               tile_spacing=(1, 1)))
        image.save('test'+self.id+ '.png')





if __name__ == "__main__":
    dataset='/Users/quynhdo/Downloads/mnist.pkl'
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]


    ae = DenoisingAutoEncoder(28*28,500,0.3)
  #  ae.fit(train_set_x, 20, 5,0.1, {'pair':[(0,1), (1,2)]})

    ae.fit(train_set_x, 20, 5,0.1)

