import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib notebook


n_steps=10000
delta_t=1e-3
t=delta_t*n_steps

omega_0=1
beta=1
lam=1
g_down=0.5
g_up=0
#g_up=np.exp(-beta*omega_0)*g_down


H_eff=np.array([[-(omega_0+1j*g_up)/2,lam],[lam,(omega_0-1j*g_down)/2]])
U=np.array([[1,0],[0,1]])-1j*H_eff*delta_t
psi=np.zeros([2,n_steps], dtype=complex)
psi[:,0]=[0,1]

#Run over time
for i in range(0,n_steps-1):
    p=delta_t*g_down*np.square(abs(psi[1,i]))
    e=np.random.rand()
    #print(p)
    if e > p:
        psi[:,i+1]=U.dot(psi[:,i])/np.sqrt(1-p)
    else:
        psi[:,i+1]=[1,0]
        
inv=(np.square(abs(psi[1,:]))-np.square(abs(psi[0,:])))
#plt.plot(np.linspace(0,t,num=n_steps),np.square(abs(psi[0,:])))
#plt.plot(np.linspace(0,t,num=n_steps),np.square(abs(psi[1,:])))
plt.plot(np.linspace(0,t,num=n_steps),inv)
