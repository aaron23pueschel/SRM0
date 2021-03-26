import numpy as np
from SRM0_ import SRM0
import matplotlib.pyplot as plt
import time as timer



############################################################################3
#################            NETWORK CLASS        #############################
##############################################################################


class neuron_matrix:
    def __init__(self,number_of_neurons,edge_list):
        self.edge_list = edge_list
        self.number_of_neurons = number_of_neurons
        self.neurons = []
        self.init_neurons()
        self.init_connections()
        self.first_neuron
        self.last_neuron
        self.desired_spike_trains = []
        
    def reset(self):
        SRM0.Time_ms = 0
        for i in self.neurons:
            i.reset()
    def init_neurons(self):
        for i in range(0,self.number_of_neurons):
            self.neurons.append(SRM0())
        self.first_neuron = self.neurons[0]
        self.last_neuron = self.neurons[self.number_of_neurons-1]
        
    def init_connections(self):
        for edge in self.edge_list:
            temp = []
            temp.append(self.neurons[edge[0]])
            temp.append(edge[2])
            self.neurons[edge[1]].connections.append(temp)

    def update_connections(self):
        for i in self.neurons:
            i.update_membrane_potential()
    def partial_nk(self,t):
        if t<=0:
            return 0
        return ((1/self.first_neuron.tau_2)*self.first_neuron.ahp_weight)*np.exp(-(t/self.first_neuron.tau_2))
    def partial_psp(self,t,weight):
        if t<=0:
            return 0
        return ((t-self.first_neuron.tau_1)*np.exp(-(t/self.first_neuron.tau_1)))*(weight/self.first_neuron.tau_1)
    def gradient_descent_linear_update(self,tau):
        for i in range(0,len(self.last_neuron.connections)):
            temp = 0
            for j in self.last_neuron.weights_at_specific_spike_times:
                temp+=50*self.ddE_ddw_ij(tau,j[0],j[1])
            self.last_neuron.connections[i][1]=self.last_neuron.connections[i][1]+temp
    def ddE_ddw_ij(self,tau,arrival_time,weight):
        temp = 0
        for i in range(0,len(self.last_neuron.spike_arrival_times)):
            t_l = self.last_neuron.spike_arrival_times[i]
            actual = self.last_neuron.spike_arrival_times
            desired = self.desired_spike_trains
            temp+=self.ddtk_ddw_ij(t_l,arrival_time,weight)*self.ddE_dd_tk_i(desired,actual,tau,i)
        return temp

    def ddtk_ddw_ij(self,t_l,time,weight):
        t_ij = time
        sum_psp = self.last_neuron.PSP(SRM0.Time_ms-(t_ij-t_l),weight)
        temp = 0
        for k in self.last_neuron.spike_arrival_times:
            if self.partial_nk(k-t_l) != 0:
                temp+=self.partial_nk(SRM0.Time_ms-(k-t_l))*self.ddtk_ddw_ij(k,time,weight)
        num = sum_psp+temp
        temp2 = 0
        for pairs in self.last_neuron.weights_at_specific_spike_times:
            temp2+=self.partial_psp(SRM0.Time_ms-(pairs[0]-t_l),pairs[1])
        temp3 = 0
        for k in self.last_neuron.spike_arrival_times:
            temp3+=self.partial_nk(SRM0.Time_ms-(k-t_l))
        den = temp3+temp2
        if den>0.00001:
            return num/den
        return 0
    def ddE_dd_tk_i(self,desired_spike_times,spike_arrival_times,tau,i):
        temp_sum1 = 0
        temp_sum2 = 0
        t = spike_arrival_times 
        s = desired_spike_times
        for j in range(0,len(t)):
            temp_sum1+=(t[j]*((t[j]-(t[i])-(t[i]/tau)*(t[j]+t[i])))/((t[j]+t[i])**3)*np.exp(-((t[j]+t[i]))/tau))
        for j in range(0,len(s)):
            temp_sum2+=(s[j]*((s[j]-t[i])-(t[i]/tau))*(s[j]+t[i])*np.exp(-((s[j]+t[i]))/tau))/(((s[j]+t[i])**3))
        return 2*(temp_sum1-temp_sum2)

   
    def main(self,time,delta_t,input,desired_spikes,num_in_first_layer):
        temp = input.copy()
        y1=[]
        y2 =[]
        x=[]
       
        
        for i in range (0,time):
            SRM0.Time_ms = i*delta_t 
            if len(temp)>0 and temp[0]==i:
                for j in range(0,num_in_first_layer):
                    self.neurons[j].input_spikes.append(i*delta_t)
                    self.neurons[j].spike_weights.append(1000)
                temp.pop(0)
            self.update_connections()
            if i%2000==0:
                self.gradient_descent_linear_update(1000)
        self.reset()
        temp = input
        for i in range (0,time):
            SRM0.Time_ms = i*delta_t 
            if len(temp)>0 and temp[0]==i:
                for j in range(0,num_in_first_layer):
                    self.neurons[j].input_spikes.append(i*delta_t)
                    self.neurons[j].spike_weights.append(100)
                    if len(temp)>0:
                        temp.pop(0)
            self.update_connections()
            y1.append(u.last_neuron.membrane_potential)
            y2.append(u.last_neuron.membrane_potential)
            x.append(i*delta_t)


        #fig, axs = plt.subplots(2)
        #axs[0].plot
        #axs[0].plot(x, y1, 'tab:orange')
        #axs[1].plot(x, y2, 'tab:blue')
        #for ax in axs.flat:
        #    ax.set(ylim = [-80,1])
        #axs.flat[0].set_title("First Neuron")
       # axs.flat[1].set_title("Last Neuron")

        plt.plot(x,y1)
        plt.show()


    ################################################################
    ###################### GRADIENTS AND TRAINING ##################      
    ################################################################      
    
    

       


 