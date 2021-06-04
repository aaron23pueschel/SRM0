import numpy as np


##################################################################################################
#################################               SRM0 MODEL            ############################
#################################################################################################


class SRM0:
    #static variables
    Time_ms = 0
    u_rest = -70
    alpha = 1.5
    A = 0
    beta = 1
    threshold = -55
    spike_height = 0
    tau_1 = 20
    tau_2 = 1.2
    delay = 0

    def __init__(self):
        self.last_efferent_spike = -np.inf    
        self.membrane_potential = -70
        self.afferent_spikes = []       ## Incoming neurons
        self.afferent_spikes_non_reset = []
        self.afferent_spikes_grad = []
        self.spike_weights = []
        self.weights = []
        self.weight = 0
        self.delays = []
        self.in_spike = False
        self.all_efferent_spikes = []   ## Outgoinng neurons
        
    def reset(self):
        self.last_efferent_spike = -np.inf    
        self.membrane_potential = -70
        self.afferent_spikes = []       ## Incoming neurons
        self.afferent_spikes_non_reset = []
        self.afferent_spikes_grad = []
        self.spike_weights = []
        self.all_efferent_spikes = []   ## Outgoinng neurons
        
    def AHP(self,last_efferent_spike_time):
        t = last_efferent_spike_time
        if(t==np.inf or t==-np.inf or t<=0):        ## Heaviside
            return 0
        return -self.A*np.exp((-(SRM0.Time_ms-t))/self.tau_2)

   
    def PSP(self,afferent_spike_time,weight):
        t = SRM0.Time_ms-afferent_spike_time
        if(t==-np.inf or t==np.inf or t<=0):        ## Heaviside
            return 0
        return weight*(1/(self.alpha*np.sqrt(t)))*np.exp(-(self.beta*self.alpha**2)/t)*np.exp(-t/self.tau_1)

    def sum_membrane(self):
        sum = 0
        self.in_spike = False
        for i in range(0,len(self.afferent_spikes)):
            for spike in self.afferent_spikes[i]:
                sum += self.PSP(spike,self.weights[i])
        #sum += self.AHP(self.last_efferent_spike)
        membrane = sum + self.u_rest
        if membrane < self.threshold:
            self.membrane_potential = sum + self.u_rest
            return
        self.spike()
    
    def spike(self):
        self.in_spike = True
        self.membrane_potential = self.u_rest
        self.last_efferent_spike = SRM0.Time_ms
        self.all_efferent_spikes.append(SRM0.Time_ms)
        self.reset_afferent_spikes()
    def reset_afferent_spikes(self):
        for i in range(0,len(self.afferent_spikes)):
            self.afferent_spikes[i].clear()
    def reset_afferent_spikes2(self):
        for i in range(0,len(self.afferent_spikes)):
            self.afferent_spikes_grad[i].clear()
        


            
            
            




