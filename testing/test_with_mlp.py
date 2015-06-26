from numpy import *
import cPickle
from random import randrange
import numpy as np
from network import *
from nnet import *


def read_cifar_file(fn):
    fo = open(fn, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def loadCifar(batchNum):
    #  For training_batches specify numbers 1 to 5
    #  for the test set pass 6
    if batchNum <= 5:
        file_name = cifar_dir + '/data_batch_' + str(batchNum)
        file_id = open(file_name, 'rb')
        dict = cPickle.load(file_id)
        file_id.close()
        return dict['data'], dict['labels']
    elif batchNum == 6:
        file_name = cifar_dir + '/test_batch'
        file_id = open(file_name, 'rb')
        dict = cPickle.load(file_id)
        file_id.close()
        return dict['data'], dict['labels']
    else:  # here we will get the whole 50,000x3072 dataset
        I = 0
        file_name = cifar_dir + '/data_batch_' + str(I + 1)
        file_id = open(file_name, 'rb')
        dict = cPickle.load(file_id)
        file_id.close()
        data = dict['data']
        labels = dict['labels']
        for I in range(1, 5):
            file_name = cifar_dir + '/data_batch_' + str(I + 1)
            file_id = open(file_name, 'rb')
            dict = cPickle.load(file_id)
            file_id.close()
            data = np.concatenate((data, dict['data']), axis=0)
            labels = np.concatenate((labels, dict['labels']), axis=0)
        return data, labels



nnet_size = np.array([[992, 500], [500, 500], [500, 10]])
learning_rate = 0.1

num_layers = 3
mlp = nnet(nnet_size, num_layers, learning_rate)

epoch = 3
cifar_dir = '/home/habtegebrial/Desktop/python-destin/cifar-10-batches-py/'
trainData = np.array([])
for I in range(0, 50000, 1):
#for I in range(499, 500, 500):
    Name = '/home/habtegebrial/GSoc/python-destin/testing/train/' + str(I + 1) + '.txt'
    file_id = open(Name, 'r')
    #print np.shape(np.ravel(np.loadtxt(Name)))
    Temp = np.array(cPickle.load(file_id))
    print np.shape(Temp)
    file_id.close()
    trainData = np.hstack((trainData, Temp))


print np.shape(trainData)
exit(0)
# cifar_dir = '/home/eskender/Destin/cifar-10-batches-py/'
#  Contains loading cifar batches and
#  feeding input to lower layer nodes


print "Training Began"
[data, y_train] = loadCifar(10)
del data

mini_batch_size = 1
num_of_mini_batches = 50000

for epoch in range(5):
	for mini in range(num_of_mini_batches):
		if mini%1000 == 0:
			print("Batch Number: %d") % (mini + 1)
		#print("Batch Number: %d") % (mini + 1)
		batch_start = mini*mini_batch_size
		batch_end = batch_start + mini_batch_size
		for img in range(batch_start, batch_end):
			target = np.zeros((num_of_classes, 1))
			target[y_train[img]] = 1
			dummy = mlp.forward(x_train[img, :])
			mlp.backward(target)
		# Update the Delta, just dividing Accumulated Delta by number of images in the batch
		mlp.average_delta_acc(mini_batch_size)
		for L in range(mlp.num_of_layers):
				mlp.layers[L].weight -= (mlp.learning_rate * mlp.layers[L].delta_acc)
		# Re-initialize the accumulated delta to zeros: for the next mini_batch
		mlp.reset_delta()

