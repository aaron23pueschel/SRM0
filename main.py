from SRM0_ import SRM0
from Network import neuron_matrix
from Master_network import Master_network_
import numpy as np
import math

delta_t = .01
time = 10000             ### Time in milliseconds is delta_t * time
input = []
#for i in range(0,int((time/100))):
  #  input.append(np.random.randint(1,time)*delta_t)

#input.sort()    ##input times in millisecond
input = [20,21,22,23,24,25,30,31,32,33,40,70]
#input = [20,20.05,20.06,25,26,27]
witness = neuron_matrix(3,input,True)
#print(witness.output_neuron.weights)

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


Master_network_(3,4,[])