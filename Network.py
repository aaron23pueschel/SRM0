import numpy as np
from SRM0_ import SRM0
import matplotlib.pyplot as plt
import time as timer
import math
from scipy.misc import derivative



############################################################################3
#################            NETWORK CLASS        #############################
##############################################################################


class neuron_matrix:
    def __init__(self,number_of_neurons,input_vec,linear):
        self.number_of_neurons = number_of_neurons
        self.all_afferent_spikes_non_reset = []
        self.all_afferent_spikes = []
        self.input_vec = input_vec
        self.desired_vec = []
        self.SRM0_list = []
        self.gradient_count = 0
        self.output_neuron = SRM0()
        if linear:
            self.init_linear()
        #else:
           # self.init_nonlinear()
        self.learn = False
        
        self.learning_rate = 0
        self.tau_gradient_descent = 0
        
        
    def init_linear(self):
        for i in range(0,self.number_of_neurons):
            rand_weight = np.random.randint(15,30)
            self.output_neuron.weights.append(rand_weight)
            rand_delay  = np.random.randint(4,9)/10
            temp_vec = []
            self.output_neuron.delays.append(rand_delay)
            for j in range(0,len(self.input_vec)):
                temp_vec.append(self.input_vec[j]+rand_delay)
            self.output_neuron.afferent_spikes.append([])
            self.output_neuron.afferent_spikes_non_reset.append([])
            self.output_neuron.afferent_spikes_grad.append([])
            self.output_neuron.spike_weights.append([])
            self.all_afferent_spikes.append(temp_vec)
            self.all_afferent_spikes_non_reset.append(temp_vec.copy())
    def init_nonlinear(self,total_neurons,neurons_per_layer):
        
        for i in range(0,total_neurons):
            new_neuron = SRM0()
            for j in range(0,neurons_per_layer):
                new_neuron.afferent_spikes.append([])
                new_neuron.afferent_spikes_non_reset.append([])
                new_neuron.afferent_spikes_grad.append([])
                new_neuron.spike_weights.append([])
                rand_weight = np.random.randint(15,30)
                new_neuron.weights.append(rand_weight)
                new_neuron.weight = rand_weight
                rand_delay  = np.random.randint(4,9)/10
                new_neuron.delay = rand_delay
                temp_vec = []
                new_neuron.delays.append(rand_delay)
            self.SRM0_list.append(new_neuron)
        for k in range(0,neurons_per_layer):
            self.output_neuron.afferent_spikes.append([])
            self.output_neuron.afferent_spikes_non_reset.append([])
            self.output_neuron.afferent_spikes_grad.append([])
            self.output_neuron.spike_weights.append([])
            rand_weight = np.random.randint(15,30)
            self.output_neuron.weights.append(rand_weight)
            self.output_neuron.weight = rand_weight
            rand_delay  = np.random.randint(4,9)/10
            self.output_neuron.delay = rand_delay
            temp_vec = []
            self.output_neuron.delays.append(rand_delay)
        self.SRM0_list.append(self.output_neuron)
    
        
    def gradient_descent_linear_update(self):    
        sum = 0
        tau = self.tau_gradient_descent
        learning_rate = self.learning_rate
        for synapses in range(0,len(self.output_neuron.afferent_spikes_non_reset)):
            for spikes in range(0,len(self.output_neuron.afferent_spikes_non_reset[synapses])):
                sum += learning_rate*self.ddE_ddw_ij(tau,self.output_neuron.afferent_spikes_non_reset[synapses][spikes],self.output_neuron.spike_weights[synapses][spikes])
            self.output_neuron.weights[synapses] = self.output_neuron.weights[synapses] - sum
            
    
    def gradient_descent_nonlinear(self):
        return 0
    def partial_tl_partial_t_ij(self,t_l,time,weight,efferent_spikes):
       
        t_ij = time
        sum_psp = self.partial_psp(SRM0.Time_ms-(t_ij-t_l),weight)
        if sum_psp==0:
            return 0
        for synapse in range(0,len(self.output_neuron.afferent_spikes_grad)):
            for afferent_spikes in range(0,len(self.output_neuron.afferent_spikes_grad[synapse])):
                s_ij = self.output_neuron.afferent_spikes_grad[synapse][afferent_spikes]
                sum1 += self.partial_psp((SRM0.Time_ms-(s_ij-t_l)),self.output_neuron.spike_weights[synapse][afferent_spikes])
        if sum1 ==0:
            print("Divide by zero error here")
            return 0
        return sum_psp/sum1


    def update_connections(self,learn):
        spike_arrival_at_soma = False
        for synapses in range(0,len(self.all_afferent_spikes)):
                while len(self.all_afferent_spikes[synapses])>0 and self.all_afferent_spikes[synapses][0]<= SRM0.Time_ms:
                    self.output_neuron.afferent_spikes[synapses].append(self.all_afferent_spikes[synapses][0])
                    self.output_neuron.afferent_spikes_grad[synapses].append(self.all_afferent_spikes[synapses][0])
                    self.output_neuron.afferent_spikes_non_reset[synapses].append(self.all_afferent_spikes[synapses][0])
                    self.output_neuron.spike_weights[synapses].append(self.output_neuron.weights[synapses])
                    #self.gradient_descent_linear_update()
                    self.all_afferent_spikes[synapses].pop(0)
                    spike_arrival_at_soma = True
        self.output_neuron.sum_membrane()
    def update_connections_nonlinear(self,i,spikes):
        spike_arrival_at_soma = False
        for synapses in range(0,len(spikes)):
                while len(spikes[synapses])>0 and spikes[synapses][0]<= SRM0.Time_ms:
                    self.SRM0_list[i].afferent_spikes[synapses].append(spikes[synapses][0])
                    self.SRM0_list[i].afferent_spikes_grad[synapses].append(spikes[synapses][0])
                    self.SRM0_list[i].afferent_spikes_non_reset[synapses].append(spikes[synapses][0])
                    self.SRM0_list[i].spike_weights[synapses].append(self.SRM0_list[i].weight)
                    spikes[synapses].pop(0)
                    spike_arrival_at_soma = True
        self.SRM0_list[i].sum_membrane()
        return spikes

    def ddE_ddw_ij(self,tau,arrival_time,weight):   
        temp = 0
        for i in range(0,len(self.output_neuron.all_efferent_spikes)):
            t_l = self.output_neuron.all_efferent_spikes[i]
            actual = self.output_neuron.all_efferent_spikes
            desired = self.desired_vec.copy()
            temp += self.partialE_partial_tk(desired,actual,tau,i)*self.ddtk_ddw_ij(t_l,arrival_time,weight,self.output_neuron.all_efferent_spikes)
        return temp                 
    def reset(self,reset_weights,input_vec):
        SRM0.Time_ms = 0
        self.all_afferent_spikes.clear()
        self.all_afferent_spikes_non_reset.clear()
        
        self.output_neuron.reset()
        if reset_weights:
            #self.output_neuron.weights.clear()
            for i in range(0,len(self.output_neuron.weights)):
                rand_weight = (np.random.randint(0,20)-10)/10
                self.output_neuron.weights[i]+=rand_weight
        for i in range(0,self.number_of_neurons):

            temp_vec = []
            self.output_neuron.afferent_spikes.append([])
            self.output_neuron.afferent_spikes_non_reset.append([])
            self.output_neuron.afferent_spikes_grad.append([])
            self.output_neuron.spike_weights.append([])
           # if reset_weights:
              #  rand_weight = np.random.randint(30,50)
               # self.output_neuron.weights.append(rand_weight)
            for j in range(0,len(input_vec)):
                temp_vec.append(input_vec[j]) #+self.output_neuron.delays[i])
            self.all_afferent_spikes.append(temp_vec)
            self.all_afferent_spikes_non_reset.append(temp_vec.copy())

    def partialE_partial_tk(self,desired_spike_times,spike_output_times,tau,i):
        
        temp_sum1 = 0
        temp_sum2 = 0
        t = spike_output_times
        s = desired_spike_times
        r = min(len(t),len(s))
        
        for j in range(0,len(t)):
            temp_sum1+=(t[j]*((t[j]-t[i])-(t[i]/tau)*(t[j]+t[i])))*np.exp(-(t[j]+t[i])/tau)/((t[j]+t[i])**3)
        for j in range(0,len(s)):
            if (s[j]-SRM0.Time_ms)<.001:
                temp_sum2+=(s[j]*((s[j]-t[i])-(t[i]/tau)*(s[j]+t[i])))*np.exp(-(s[j]+t[i])/tau)/((s[j]+t[i])**3)
            else:
                break
        x = (2*(temp_sum1-temp_sum2))
        return 2*(temp_sum1-temp_sum2)

    def PSP(self,t,weight):
        if(t==-np.inf or t==np.inf or t<=0):        
            return 0
        return weight*(1/(SRM0.alpha*np.sqrt(t)))*np.exp(-(SRM0.beta*SRM0.alpha**2)/t)*np.exp(-t/SRM0.tau_1)
    def ddtk_ddw_ij(self,t_l,time,weight,efferent_spikes):
       
        t_ij = time
        sum_psp = self.PSP(SRM0.Time_ms-(t_ij-t_l),weight)
       
        sum = 0
        #efferent_spikes_ = efferent_spikes.copy()
        #for spikes in efferent_spikes_:
          #  efferent_spikes_.pop(len(efferent_spikes_)-1)
          #  sum += self.partial_nk(spikes - t_l)#*self.ddtk_ddw_ij(t_l,time,weight,efferent_spikes_)
        num = sum_psp  
        sum1 = 0
        for synapse in range(0,len(self.output_neuron.afferent_spikes_grad)):
            for afferent_spikes in range(0,len(self.output_neuron.afferent_spikes_grad[synapse])):
                s_ij = self.output_neuron.afferent_spikes_grad[synapse][afferent_spikes]
                sum1 += self.partial_psp((SRM0.Time_ms-(s_ij-t_l)),self.output_neuron.spike_weights[synapse][afferent_spikes])
        sum2 = 0
        #if(sum1<0):
           # print("here")
        #for efferent_spikes in self.output_neuron.all_efferent_spikes:
            #sum2 += self.partial_nk(efferent_spikes-t_l)
        den = sum1 + sum2
        #print(sum1)
        if den!=0:
            x = (num/den)
            return num/den
            
        return 0
    def partial_psp(self,t,weight):
        if(t<0):
            return 0
        #def f(t):
            #return self.PSP(t,weight)
        #return derivative(f, t, dx=1e-6)
        return(-((2*t**2+SRM0.tau_1*t-2*SRM0.alpha**2*SRM0.beta*SRM0.tau_1)*np.exp(-t/SRM0.tau_1-(SRM0.alpha**2*SRM0.beta)/t))/(2*SRM0.alpha*SRM0.tau_1*np.sqrt(math.pow(t,5))))

    def main(self,time,delta_t,learn,desired_spikes):
        x = []
        y = []
        self.learning_rate = .05
        self.tau_gradient_descent = 150
        temp = []
        self.desired_vec = desired_spikes
        random_update_time = []
        first = True
        #if learn:
          #  self.learn = True
            
         #   if first:
            #    for i in range(0,int((time*delta_t/5))):
            #        random_update_time.append(np.random.randint(0,int(time)*delta_t))
            #        first = False
            #    random_update_time.sort()
        
        for T in range(0,time):
            SRM0.Time_ms = delta_t*T
            self.update_connections(learn)
            y.append(self.output_neuron.membrane_potential)
            x.append(SRM0.Time_ms)
            if(self.output_neuron.in_spike and learn):
                self.gradient_descent_linear_update()
                self.output_neuron.reset_afferent_spikes2()
                #self.output_neuron.afferent_spikes_non_reset.pop(0)
                #self.output_neuron.all_efferent_spikes.pop(0)
        #print(self.output_neuron.weights,": Neuron weights before learning ")  
        
        #print(self.output_neuron.weights,": Neuron weights after learning ")
        #print(self.output_neuron.weights)
        #print(self.output_neuron.all_efferent_spikes)
        #print(self.output_neuron.all_efferent_spikes)
        #plt.plot(x,y)
        #plt.show()
       

        
        return self.output_neuron.all_efferent_spikes
        


       
       


 