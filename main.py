from SRM0_ import SRM0
from Network import neuron_matrix
import numpy as np
    ################################################################
     ###################### NETWORK GENERATION ##################      
    ################################################################      



def network_1to1(layers,neurons_per_layer):
    temp = []
    for i in range(1,neurons_per_layer+1):
        rand = np.random.randint(-10,40)/10
        temp.append([0,i,rand])
    for k in range(0,layers-1):
        for j in range(1,neurons_per_layer+1):
            for i in range(1,neurons_per_layer+1):
                rand = np.random.randint(-10,40)/10
                temp.append([(k*neurons_per_layer)+j,(k*neurons_per_layer)+neurons_per_layer+i,rand])
    for i in range(0,neurons_per_layer):
        rand = np.random.randint(-10,40)/10
        temp.append([neurons_per_layer*(layers-1)+1+i,neurons_per_layer*layers+1,rand])
    return temp

def linear(neurons_per_layer):
    temp = []
    for i in range(0,neurons_per_layer):
        rand = np.random.randint(20,40)/10
        temp.append([i,neurons_per_layer,rand])
    print(temp)
    return temp

##########################################################################
##########################         MAIN         ##########################
##########################################################################



layers = 1
neurons_per_layer = 5
#edge_list_test = network_1to1(layers,neurons_per_layer)
edge_list_test = linear(neurons_per_layer)
u = neuron_matrix(neurons_per_layer*layers+1,edge_list_test)
delta_t = .1

time_ = 10000


input = [1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000]
for i in edge_list_test:
    print(i[2])
u.main(time_,delta_t,input,[5000],neurons_per_layer)

print(u.last_neuron.weights_at_specific_spike_times)
#print(u.first_neuron.spike_arrival_times)
#print(u.last_neuron.spike_arrival_times)
#print(u.last_neuron.spike_weights)
#print(u.last_neuron.weights_at_specific_spike_times)


u.reset()