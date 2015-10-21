__author__ = 'quynhdo'


class NNNetComplex:
    def __init__(self, *components, *connection_layer_ids, cost_function):
        self.components=components
        self.connection_layer_ids = connection_layer_ids
        self.layers=[]
        self.connections=[]
        self.params=[]

        lst=[]
        i=0
        for i in range(len(components)):
            for j in range(connection_layer_ids[i]):
                self.layers.append(components[i].layers[j])
                i+=1
                self.connections.append(components[i].connections[j])
            self.layers.append(components[i].layers[connection_layer_ids[i]])
            lst.append(i)
            i+=1

        for l in lst:





        self.cost_function= cost_function


