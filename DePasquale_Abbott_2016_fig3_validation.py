 %reset -f

import numpy as np
import matplotlib.pyplot as plt
import time 
from numba import jit
import os

h = 1 #ms
n_trials_load = 50

os.chdir(r'E:\20191019_Michigan_Comp_NSC\DePasquale_Abbott_2016_arXiv')
J   = np.load(('fig3_h' + str(h*1000) + 'us_' + str(n_trials_load) + '_J.npy'))
J_f = np.load(('fig3_h' + str(h*1000) + 'us_' + str(n_trials_load) + '_J_f.npy'))
U   = np.load(('fig3_h' + str(h*1000) + 'us_' + str(n_trials_load) + '_U.npy'))
W   = np.load(('fig3_h' + str(h*1000) + 'us_' + str(n_trials_load) + '_W.npy'))

n_trials = 5

# seq type, in, out
# 1, F F, F
# 2, F T, T
# 3, T F, T
# 4, T T, F

# @jit(nopython=True) 
def F_in_out(h,n_trials):
    inter_trial_min = 1000 #ms
    inter_trial_max = 5000 #ms

    ones100 = np.ones(int(np.floor(100/h)))*0.3
    ones300 = np.ones(int(np.floor(300/h)))*0.3
    zeros100 = np.zeros(int(np.floor(100/h)))
    zeros300 = np.zeros(int(np.floor(300/h)))
    zeros500 = np.zeros(int(np.floor((500)/h)))
    sinpos = np.sin((2*np.pi)*np.arange(0,0.5,(h/1000)))
    sinneg = -np.sin((2*np.pi)*np.arange(0,0.5,(h/1000)))
    if np.shape(sinpos) > np.shape(zeros500):
        sinpos = sinpos[0:-1]
        sinneg = sinneg[0:-1]
    elif np.shape(sinpos) < np.shape(zeros500):
        sinpos = np.hstack((sinpos,np.array([0])))
        sinneg = np.hstack((sinneg,np.array([0])))
    ini_seg = int(np.floor((inter_trial_min/h)*np.random.rand()))
    inter_trial = (np.floor((inter_trial_min+inter_trial_max*np.random.rand(n_trials))/h)).astype(np.int64)
    seq_type = np.random.randint(1,5,(n_trials))
    F_in_shape = ini_seg; F_out_shape = ini_seg; 
    for tr in np.arange(n_trials):
        if seq_type[tr] == 1:
           F_in_shape   = F_in_shape  + int((100+300+100+300+500)/h) + inter_trial[tr]
           F_out_shape  = F_out_shape + int((100+300+100+300+500)/h) + inter_trial[tr]
        elif seq_type[tr] == 2:
           F_in_shape   = F_in_shape  + int((100+300+300+300+500)/h) + inter_trial[tr]
           F_out_shape  = F_out_shape + int((100+300+300+300+500)/h) + inter_trial[tr]
        elif seq_type[tr] == 3:
           F_in_shape   = F_in_shape  + int((300+300+100+300+500)/h) + inter_trial[tr]
           F_out_shape  = F_out_shape + int((300+300+100+300+500)/h) + inter_trial[tr]
        elif seq_type[tr] == 4:
           F_in_shape   = F_in_shape  + int((300+300+300+300+500)/h) + inter_trial[tr]
           F_out_shape  = F_out_shape + int((300+300+300+300+500)/h) + inter_trial[tr]
    F_in = np.zeros(F_in_shape); F_out = np.zeros(F_out_shape); 
    tr_onset = ini_seg
    for tr in np.arange(n_trials):
        zerosinter = np.zeros(inter_trial[tr])
        if seq_type[tr] == 1:
           tr_duration = np.shape(np.hstack((ones100,zeros300,ones100,zeros300,zeros500,zerosinter)))[0]
           F_in[tr_onset:(tr_onset+tr_duration)]   = np.hstack((ones100,zeros300,ones100,zeros300,zeros500,zerosinter))
           F_out[tr_onset:(tr_onset+tr_duration)]  = np.hstack((zeros100,zeros300,zeros100,zeros300,sinneg,zerosinter))
           tr_onset = tr_onset + tr_duration
        elif seq_type[tr] == 2:
           tr_duration = np.shape(np.hstack((ones100,zeros300,ones300,zeros300,zeros500,zerosinter)))[0]
           F_in[tr_onset:(tr_onset+tr_duration)]   = np.hstack((ones100,zeros300,ones300,zeros300,zeros500,zerosinter))
           F_out[tr_onset:(tr_onset+tr_duration)]  = np.hstack((zeros100,zeros300,zeros300,zeros300,sinpos,zerosinter))
           tr_onset = tr_onset + tr_duration
        elif seq_type[tr] == 3:
           tr_duration = np.shape(np.hstack((ones300,zeros300,ones100,zeros300,zeros500,zerosinter)))[0]
           F_in[tr_onset:(tr_onset+tr_duration)]   = np.hstack((ones300,zeros300,ones100,zeros300,zeros500,zerosinter))
           F_out[tr_onset:(tr_onset+tr_duration)]  = np.hstack((zeros300,zeros300,zeros100,zeros300,sinpos,zerosinter))
           tr_onset = tr_onset + tr_duration
        elif seq_type[tr] == 4:
           tr_duration = np.shape(np.hstack((ones300,zeros300,ones300,zeros300,zeros500,zerosinter)))[0]
           F_in[tr_onset:(tr_onset+tr_duration)]   = np.hstack((ones300,zeros300,ones300,zeros300,zeros500,zerosinter))
           F_out[tr_onset:(tr_onset+tr_duration)]  = np.hstack((zeros300,zeros300,zeros300,zeros300,sinneg,zerosinter))
           tr_onset = tr_onset + tr_duration
        #print(tr)
    return F_in, F_out

start_time = time.time()
F_in, F_out = F_in_out(h,n_trials)
print("--- %s seconds ---" % (time.time() - start_time))

# In[ ] spiking network, validate

T = np.shape(F_in)[0]*h
print(T)

# @jit(nopython=True)
def solve_W(h,F_in,J,U,J_f):
    T = np.shape(F_in)[0]*h #ms
    t = np.arange(0,T,h)
    
    N = 3000 
    N_out = 1
    N_in  = 1
    mu = -40 
    g_f = 12 
    g = 10  
    I = 10
    tau_s = 100
    tau_f = 2
    tau_m = 20
    V_rest = -65
    V_th = -55
    V = np.random.uniform(V_rest,V_rest+10,(N,1))
    s = np.random.uniform(0,1,(N,1))
    f = np.random.uniform(0,1,(N,1))
    
    s_time = np.zeros((N,np.shape(t)[0]))
    firings = np.zeros((N,np.shape(t)[0]))
    for i in np.arange(np.shape(t)[0]):
        fired = np.nonzero(V >= V_th-0.5)[0] ####
        firings[fired,i] = 1
        V[fired]=V_rest
        s[fired]=s[fired]+1
        f[fired]=f[fired]+1
        V = V + h*((V_rest-V+g*(J@s+J_f@f+U*F_in[i])+I)/tau_m)
        s = s + h*(-s/tau_s)
        f = f + h*(-f/tau_f)
        s_time[:,i] = s.ravel()
        if np.remainder(t[i],1000) == 0:
            print(t[i])
    return s_time, firings

start_time = time.time()
s_time, firings = solve_W(h,F_in,J,U,J_f)
print("--- %s seconds ---" % (time.time() - start_time))

Ws = (s_time.transpose())@W
# In[] save
os.chdir(r'E:\20191019_Michigan_Comp_NSC\DePasquale_Abbott_2016_arXiv')
np.save(('fig3_h' + str(h*1000) + 'us_' + str(n_trials_load) + '_Ws_validation_' + str(n_trials) + 'trials'),Ws) 


# In[] plot
'''
#  %matplotlib auto
plt.plot(Ws)
plt.plot(F_out)
'''



