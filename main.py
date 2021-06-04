from SRM0_ import SRM0
from Network import neuron_matrix
from Master_network import Master_network_
import numpy as np
import math
import matplotlib.pyplot as plt

delta_t = .01
time = 10000             ### Time in milliseconds is delta_t * time
input = []
#for i in range(0,int((time/100))):
  #  input.append(np.random.randint(1,time)*delta_t)

#input.sort()    ##input times in millisecond
input = [20,21,22,22.5,22.7,23.4,23,24.5,24.7,25,25.5,26,27,28,30,31,32,33,40,70]
#input = [20,20.05,20.06,25,26,27]
#witness = neuron_matrix(10,input,True)
#print(witness.output_neuron.weights)
network = Master_network_(2,2,input)
#witness.output_neuron.weights = [130]
#print("Witness weights: ",witness.output_neuron.weights)
#witness_output = witness.main(time,delta_t,False,[]) 
#print(witness.output_neuron.all_efferent_spikes)
#witness.reset(True,input)
#weights_before = witness.output_neuron.weights
#print(weights_before)
#witness.output_neuron.weights = [140]
#before = witness.main(time,delta_t,True,witness_output)
#print("Witness output: ",(witness_output),"before learning: ", (before))
#print("Before learning weights: ", witness.output_neuron.weights)
#after = [] 
#for i in range(0,1000): 
  #  witness.reset(False,input)
  #  after = witness.main(time,delta_t,True,witness_output)
  ##  print(witness.output_neuron.weights)
  #  print(witness.output_neuron.all_efferent_spikes)
#print("Witness output: ",witness_output)
#print("after: ",after)
#print("before: ",before)
#print(weights_before)
#print(witness.output_neuron.weights)
#print("After learning weights: ", witness.output_neuron.weights)
#after = witness.main(time,delta_t,False,[])
time = []
test1,test2,test3,test4,test5 = [],[],[],[],[]
for i in range(0,10000):
    network.update_neurons()
    SRM0.Time_ms+=.01

    time.append(SRM0.Time_ms)
   
    test1.append(network.matrix.SRM0_list[0].membrane_potential)
    test2.append(network.matrix.SRM0_list[1].membrane_potential)
    test3.append(network.matrix.SRM0_list[2].membrane_potential)
    test4.append(network.matrix.SRM0_list[3].membrane_potential)
    test5.append(network.matrix.SRM0_list[4].membrane_potential)
count = 0

fig, axs = plt.subplots(3, 2)
x = time
axs[0, 0].plot(x, test1)
axs[0, 1].plot(x, test2, 'tab:orange')
axs[1, 0].plot(x, test3, 'tab:green')
axs[1, 1].plot(x, test4, 'tab:red')
axs[2, 0].plot(x, test5, 'tab:green')
axs[2, 1].plot(x, x, 'tab:red')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()