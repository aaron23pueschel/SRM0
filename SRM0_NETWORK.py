import numpy as np
import matplotlib.pyplot as plt



##################################################################################################
#################################               SRM0 MODEL            ############################
#################################################################################################


class SRM0:
    #static variables
    Time_ms = 0
    u_rest = -70
    threshold = -55
    spike_height = 0
    tau_1 = 20
    tau_2 = 1.2
    delay = 0

    def __init__(self):
        self.psp_weight = 1
        self.ahp_weight = 5
        self.last_spike = -np.inf    
        self.membrane_potential = self.u_rest
        self.input_spikes = [-np.inf]
        self.connections = []
        self.spike_arrival_times = []
        self.spike_weights = [0]
        self.weights_at_specific_spike_times = []
        self.in_spike = False



    def AHP(self,t):
        if(t==np.inf):
            return 0
        return -self.ahp_weight*np.exp((-t)/self.tau_2)


    def PSP(self,t,weight):
        if(t==-np.inf or t==np.inf or t<0):
            return 0
        return weight*(t)*np.exp((-t)/self.tau_1)


    def sum_MP(self):
        temp = 0
        for i in range(0,len(self.connections)):
            if self.connections[i][0].in_spike:
                self.input_spikes.append(self.connections[i][0].Time_ms)
                self.spike_weights.append(self.connections[i][1])
        for i in range(0,len(self.input_spikes)):
            if self.input_spikes[i]<=self.Time_ms:
                temp+=self.PSP(self.Time_ms-self.input_spikes[i],self.spike_weights[i])
        if temp+self.AHP(self.Time_ms-self.last_spike)+self.u_rest>self.threshold:
            self.spike()
            self.in_spike = True
        else:
            self.membrane_potential = temp+self.AHP(self.Time_ms-self.last_spike)+self.u_rest
            self.in_spike = False    

    def update_membrane_potential(self):
        self.sum_MP()
       

    def spike(self):
            self.membrane_potential = self.spike_height
            self.last_spike = self.Time_ms
            self.input_spikes = []
            self.spike_weights = []
            self.spike_arrival_times.append(self.Time_ms)
    def spike_reset(self):
            self.membrane_potential = self.u_rest
            
            
       
    def update_PSP_weight(self,new_weight):
        self.psp_weight = new_weight

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


    ################################################################
    ###################### GRADIENTS AND TRAINING ##################      
    ################################################################      
    
    


def psp_prime():
    return 0
def ahp_prime():
    return 0
def psp():
    return 0
def ahp():
    return 0
                
def partialE_partial_tk_i(desired_spike_times,neuron,tau,i):
    temp_sum1 = 0
    temp_sum2 = 0
    t = neuron.spike_arrival_times 
    s = desired_spike_times
    for j in range(0,len(neuron.spike_arrival_times)):
        temp_sum1+=(t[j]*((t[j]-(t[i])-(t[i]/tau)*(t[j]+t[i])))/((t[j]+t[i])**3)*np.exp(-((t[j]+t[i]))/tau))
    for j in range(0,len(desired_spike_times)):
        temp_sum2+=(t[j]*((t[j]-(s[i])-(t[i]/tau)*(s[j]+t[i])))/((s[j]+t[i])**3)*np.exp(-((s[j]+t[i]))/tau))
    return 2*(temp_sum1-temp_sum2)




def partialt_partialW(i,j,output_spike_times,incoming_spike_times,weights,t_l,weight_at_psp_i):
    s = output_spike_times
    t = incoming_spike_times
    psp(t[i][j]-t_l,weight_at_psp_i)

    
       


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




##########################################################################
##########################         MAIN         ##########################
##########################################################################



layers = 2
neurons_per_layer = 10
edge_list_test = network_1to1(layers,neurons_per_layer)
u = neuron_matrix(neurons_per_layer*layers+2,edge_list_test)
delta_t = .1

y1= []
y2 = []
xn = []
for i in range (0,3000):
    SRM0.Time_ms = i*delta_t 
    if i==200 or i==400 or i==600 or i==800:
        u.neurons[0].input_spikes.append(i*delta_t)
        u.neurons[0].spike_weights.append(3000)
    u.update_connections()

    y1.append(u.first_neuron.membrane_potential)
    y2.append(u.last_neuron.membrane_potential)
    xn.append(i*delta_t)

fig, axs = plt.subplots(2)
axs[0].plot
axs[0].plot(xn, y1, 'tab:orange')
axs[1].plot(xn, y2, 'tab:blue')
for ax in axs.flat:
    ax.set(ylim = [-80,1])
axs.flat[0].set_title("First Neuron")
axs.flat[1].set_title("Last Neuron")

plt.show()
