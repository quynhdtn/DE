__author__ = 'quynhdo'
import theano as th
from liir.nlp.ml.nn.base.AutoEncoder import AutoEncoder
from liir.nlp.ml.classifiers.linear.logistic import load_data
try:
    import PIL.Image as Image
except ImportError:
    import Image
from liir.nlp.ml.classifiers.linear.utils import tile_raster_images
from liir.nlp.ml.nn.base.Layer  import Layer
from liir.nlp.ml.nn.base.NNNet  import NNNet
from liir.nlp.ml.nn.base.Functions import CrossEntroyCostFunction
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self,nIn=700, nHidden=500, corruption_level=0.1):

        self.corruption_level = corruption_level


        self.rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))



        ilayer = Layer(numNodes=nIn, ltype = Layer.Layer_Type_Input, id="0", input_process_func=MyDenoisingProcessFunction)
        hlayer = Layer(numNodes=nHidden, ltype = Layer.Layer_Type_Hidden, id="1")
        olayer = Layer(numNodes=nIn, ltype = Layer.Layer_Type_Output, id="2")

        ilayer.input_process_func.

        # declare nnnet
        self.net = NNNet(ilayer, hlayer, olayer, cost_function=CrossEntroyCostFunction)

        # change parameter constraint
        conn1 = self.net.connections[0]
        conn2 = self.net.connections[1]

        self.net.params.remove(conn2.W)
        conn2.W = conn1.W.T



    def MyDenoisingProcessFunction(x, corruptionlevel=self.corruption_level):  return theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level, dtype=th.config.floatX) * x


    def fit(self, train_data, batch_size, training_epochs, learning_rate):
        self.net.fit(train_data, train_data, batch_size, training_epochs, learning_rate)
        image = Image.fromarray(
            tile_raster_images(X=self.net.connections[0].W.get_value(borrow=True).T,
                               img_shape=(28, 28), tile_shape=(10, 10),
                               tile_spacing=(1, 1)))
        image.save('test_ae.png')

dataset='/Users/quynhdo/Downloads/mnist.pkl'
datasets = load_data(dataset)
train_set_x, train_set_y = datasets[0]


ae = DenoisingAutoEncoder(28*28,500,0.3)
ae.fit(train_set_x, 20, 5,0.1)
