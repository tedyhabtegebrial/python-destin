import cPickle as pickle
from network import *
from time import time
# Load Network
with open('DESTIN.pkl', 'rb') as input:
	DESTIN = pickle.load(input)
# training or not
DESTIN.setmode(False)
DESTIN.set_lowest_layer(1)
print DESTIN.number_of_layers
print np.shape(DESTIN.layers[0][2].nodes[0][0].belief)
print np.shape(DESTIN.network_belief['belief'])
# exit(0)
# Load Data
[data, labels] = loadCifar(10)

t = time()
for epoches in range(1):
    print "Epoches = " + str(epoches)
    for I in range(data.shape[0]):
        if I % 500 == 0:
            print("Training Iteration Number %d" % I)
        for L in range(DESTIN.number_of_layers):
            if L == 0:
                img = data[I][:].reshape(32, 32, 3)
                # img = 0.5*img[:,:,0] + 0.4*img[:,:,1] + 0.1*img[:,:,2]
                # Load data functionality From the Old Implementation
                DESTIN.layers[0][L].load_input(img, [4, 4])
                layer_features = DESTIN.extract_feature(L, [2,2], 'avg')
                #print np.shape(layer_features)
                feature_to_store = layer_features
                DESTIN.apply_soft_max(L)
            else:
                #extract_feature(self, layer_num, dims, pooling_type='avg')
                DESTIN.load_layer_data(layer_features, L)
                layer_features = DESTIN.extract_feature(L, [2,2], 'avg')
                #DESTIN.apply_soft_max(L)
		#DESTIN.update_belief_exporter_layer_one_inputs()
		#print np.shape(DESTIN.network_belief['belief'])
		Name = 'train/' + str(I + 1) + '.txt'
		file_id = open(Name, 'w')
		pickle.dump(np.array(feature_to_store), file_id)
		file_id.close()
		DESTIN.clean_belief_exporter()
		#print np.shape(DESTIN.network_belief['belief'])
		#exit(0)
		#exit(0)
		"""
		if I in range(499, 50499, 500):
			print np.shape(DESTIN.network_belief['belief'])
			Name = 'train/' + str(I + 1) + '.txt'
			file_id = open(Name, 'w')
			pickle.dump(np.array(DESTIN.network_belief['belief']), file_id)
			file_id.close()
			# Get rid-off accumulated training beliefs
			DESTIN.clean_belief_exporter()
		"""
