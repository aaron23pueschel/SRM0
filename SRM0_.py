import numpy as np


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
        self.ahp_weight = 10
        self.last_spike = -np.inf    
        self.membrane_potential = self.u_rest
        self.input_spikes = [-np.inf]
        self.connections = []
        self.spike_arrival_times = []
        self.spike_weights = [0]
        self.weights_at_specific_spike_times = []
        self.in_spike = False
    def reset(self):
        self.psp_weight = 1
        self.ahp_weight = 10
        self.last_spike = -np.inf    
        self.membrane_potential = self.u_rest
        self.input_spikes = [-np.inf]
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
                self.weights_at_specific_spike_times.append([self.connections[i][0].Time_ms,self.connections[i][1]])
        for i in range(0,len(self.input_spikes)):
            if self.input_spikes[i]<=self.Time_ms:
                temp+=self.PSP(self.Time_ms-self.input_spikes[i],self.spike_weights[i])
            else:
                break
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
            self.input_spikes.clear()
            self.spike_weights.clear()
            self.spike_arrival_times.append(self.Time_ms)
    def spike_reset(self):
            self.membrane_potential = self.u_rest
            
            
       
    def update_PSP_weight(self,new_weight):
        self.psp_weight = new_weight




