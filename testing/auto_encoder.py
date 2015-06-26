__author__ = 'teddy'

import numpy as np
import numpy.random as rand
#from network import *
from operations import *
def corrupt(x):
	level = 0.1
	length = np.size(x)
	rand = np.random.rand(1, length)
	x = np.reshape(x,(1,length))
	x[rand>(1-level)] = 0.0
	return x

class layer:

    def __init__(self, layer_dim , eps = 0.1):
        n_input = layer_dim[0]
        n_output = layer_dim[1]
        self.input_ = np.array([])
        self.activation = np.zeros((layer_dim[1], 1))
        self.weight = init_weight(layer_dim)
        self.delta_acc = np.zeros_like(self.weight)
        self.grad = np.zeros((n_output, n_input))
        self.delta = np.zeros(n_output)

#denoising auto_encoder
class d_ae:

    def __init__(self, mlp_size, learning_rate):
        inp_size = mlp_size[0]
        hid_size = mlp_size[1]
        shape = np.array([[inp_size, hid_size],[hid_size, inp_size]])
        # mlp_size is a vector holding sizes of each layer like list[[400,100],[100,10]]
        self.num_of_layers = 2
        layers = [layer(shape[j][:]) for j in range(2)]
        self.layers = layers
        self.layers[0].weight = np.transpose(self.layers[1].weight)
        self.learning_rate = learning_rate
        self.input_ = np.array([])
        self.mlp_size = mlp_size

    def forward(self, x):
        self.input_ = vectorize(x)
        input_ = self.input_
        for i in range(self.num_of_layers):
            self.layers[i].activation = vectorize(sigmoid(np.dot(self.layers[i].weight, input_) + 1)) # TODO
            input_ = self.layers[i].activation
        #self.layers[self.num_of_layers -1].activation = soft_max(self.layers[self.num_of_layers -1].activation)
        return self.layers[-1].activation
        #np.argmax(np.exp(self.layers[-1].activation)/np.sum(np.sum(np.exp(self.layers[-1].activation))))

    def backward(self, y):
        y = y.reshape(np.size(y),1)
        self.target = y
        self.target = y
        self.layers[-1].delta = self.layers[-1].activation - self.target
        self.layers[-1].delta_acc = np.dot(self.layers[-1].delta, self.layers[-2].activation.transpose())

        for i in range(self.num_of_layers-2, 0, -1):
            self.layers[i].delta = np.dot(self.layers[i+1].weight.transpose(), self.layers[i+1].delta) * \
                                   (self.layers[i].activation * (1- self.layers[i].activation))
            self.layers[i].delta_acc = np.outer(1.0 * self.layers[i].delta, 1.0 * self.layers[i-1].activation)
        self.layers[0].delta = np.dot(self.layers[1].weight.transpose(), self.layers[1].delta) * \
                                   (self.layers[0].activation * (1- self.layers[0].activation))
        self.layers[0].delta_acc = np.outer(self.layers[0].delta, self.input_)
        self.layers[0].delta_acc += self.layers[1].delta_acc.transpose()
        self.layers[1].delta_acc = self.layers[0].delta_acc.transpose()
        self.layers[1].weight -= self.learning_rate*self.layers[1].delta_acc
        self.layers[0].weight -= self.learning_rate*self.layers[0].delta_acc
