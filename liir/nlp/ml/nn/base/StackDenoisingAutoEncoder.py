from liir.nlp.ml.nn.base.Functions import SquaredErrorCostFunction, SigmoidOutputFunction, \
    NegativeLogLikelihoodCostFunction

__author__ = 'quynhdo'
from liir.nlp.ml.nn.base.DenoisingAutoEncoder import DenoisingAutoEncoder
from liir.nlp.ml.classifiers.linear.logistic import load_data
import theano as th
try:
    import PIL.Image as Image
except ImportError:
    import Image
from liir.nlp.ml.classifiers.linear.utils import tile_raster_images
from liir.nlp.ml.nn.base.MLP import MLP
from liir.nlp.ml.nn.base.Layer import Layer


class StackDenoisingAutoEncoder:


    def __init__(self, numInput, numHiddens, numOutput, corruption_level,id=""):

        self.layers=[]
        ilayer = Layer(numNodes=numInput, ltype = Layer.Layer_Type_Input, id=id+"0")  # input layer
        self.layers.append(ilayer)
        for i in  range(len(numHiddens)):
            hlayer = Layer(numNodes=numHiddens[i], ltype = Layer.Layer_Type_Hidden, id=id+str(i))
            self.layers.append(hlayer)
        self.mlp = MLP(self.layers, numOutput, activate_function=SigmoidOutputFunction, cost_function=NegativeLogLikelihoodCostFunction)

        for i in range(1, len(self.layers)):
            dA= DenoisingAutoEncoder(self.layers[i-1].size, self.layers[i].size)


    def __init__(self, *dA, size_output_layer=100,  activate_function=None, cost_function=SquaredErrorCostFunction):
        self.dAs= dA
        layers =[]
        for i in range(len(self.dAs)):
            if i==0:
                #the first layer:
                layers.append(self.dAs[0].layers[0])
                layers.append(self.dAs[0].layers[1])
            else:
                layers.append(self.dAs[i].layers[1])
        self.mlp = MLP(layers=layers,size_output_layer=size_output_layer,activate_function= activate_function, cost_function= cost_function)

    def preTrain(self, train_data, batch_size, training_epochs, learning_rate):

        ### pretrain step
        td= train_data
        for i in range(len(self.dAs)):
            print ("Pre-train model %d..." % i)
            self.dAs[i].fit(td,batch_size, training_epochs, learning_rate)
            td=th.shared(self.dAs[i].connections[0].getOutputValue(td))

        ###
        for i in range(len(self.dAs)):
            self.mlp.params.remove(self.mlp.connections[i].W)
            self.mlp.params.remove(self.mlp.connections[i].b)
            self.mlp.connections[i].W = self.dAs[i].connections[0].W
            self.mlp.connections[i].b = self.dAs[i].connections[0].b
            self.mlp.params.append(self.mlp.connections[i].W)
            self.mlp.params.append(self.mlp.connections[i].b)

        print("Finish pre-training!")

    def fit( self, train_data, train_data_label, batch_size, training_epochs, learning_rate):
        self.mlp.fit(train_data, train_data_label, batch_size, training_epochs, learning_rate)


dataset='/Users/quynhdo/Downloads/mnist.pkl'
datasets = load_data(dataset)
train_set_x, train_set_y = datasets[0]


ae1 = DenoisingAutoEncoder(28*28,400,0.1,id="0")

ae2 = DenoisingAutoEncoder(400,32*32,0.2,id="1")

ae3= DenoisingAutoEncoder(32*32,700,0.3,id="2")

sda= StackDenoisingAutoEncoder(ae1,ae2,ae3)
sda.preTrain(train_set_x, 20, 1,0.1)