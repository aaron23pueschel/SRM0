from SRM0_ import SRM0
from Network import neuron_matrix
import numpy as np
import copy
import math
class Master_network_:
    def __init__(self,layers,neurons_per_layer,input_vec):
        self.layers = layers
        self.neurons_per_layer = neurons_per_layer
        self.input_vec = input_vec
        self.all_neurons_graph = self.network_1to1(layers,neurons_per_layer)
        is_linear = True
        if(layers>1):
            is_linear = False
        self.matrix = neuron_matrix(len(self.all_neurons_graph),input_vec,is_linear)
        self.matrix.init_nonlinear(neurons_per_layer*layers,neurons_per_layer)
        self.init_afferent_spikes()
    def init_afferent_spikes(self):
        for i in range(1,self.neurons_per_layer+1):
            rand_delay  = np.random.randint(4,9)/10
            temp_vec = []
            self.matrix.SRM0_list[i].delay = (rand_delay)
            for j in range(0,len(self.input_vec)):
                temp_vec.append(self.input_vec[j]+rand_delay)
            self.matrix.all_afferent_spikes.append(temp_vec)
            self.matrix.all_afferent_spikes_non_reset.append(temp_vec.copy())
    def update_neurons(self):
        temp = [] 
        for k in range(0,self.neurons_per_layer):
            temp = self.matrix.update_connections_nonlinear(k,copy.deepcopy(self.matrix.all_afferent_spikes))
        self.matrix.all_afferent_spikes = temp
        for i in range(0,self.all_neurons_graph[(len(self.all_neurons_graph)-1)][1]):
            if(self.matrix.SRM0_list[i].in_spike):
                index = 0
                while(index<len(self.all_neurons_graph) and self.all_neurons_graph[index][0]==i):
                    self.matrix.SRM0_list[self.all_neurons_graph[index][1]-1].afferent_spikes[self.all_neurons_graph[index][0]-1].append(SRM0.Time_ms+self.matrix.SRM0_list[i].delay)
                    index += 1
            self.matrix.SRM0_list[self.all_neurons_graph[i][1]-1].sum_membrane()
    def network_1to1(self,layers,neurons_per_layer):
        temp = []
    
        for k in range(0,layers-1):
            for j in range(1,neurons_per_layer+1):
                for i in range(1,neurons_per_layer+1):
                    temp.append([(k*neurons_per_layer)+j,(k*neurons_per_layer)+neurons_per_layer+i])
        for i in range(0,neurons_per_layer):
            temp.append([neurons_per_layer*(layers-1)+1+i,neurons_per_layer*layers+1])
        return temp

