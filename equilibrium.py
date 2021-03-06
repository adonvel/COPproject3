import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import expm
%matplotlib notebook


n_steps=100000
delta_t=1e-4
t=delta_t*n_steps

omega_0=1
beta=1
lam=1
g_down=0.5
#g_up=0
g_up=np.exp(-beta*omega_0)*g_down


H_eff=np.array([[-(omega_0+1j*g_up)/2,lam],[lam,(omega_0-1j*g_down)/2]])
U=expm(-1j*H_eff*delta_t)
psi=np.zeros([2,n_steps], dtype=complex)
psi[:,0]=[0,1]

#Run over time
for i in range(0,n_steps-1):
    
    p_down=delta_t*g_down*np.square(abs(psi[1,i]))
    p_up=delta_t*g_down*np.square(abs(psi[0,i]))
    p=p_down+p_up
    e=np.random.rand()
    if e > p:
        psi[:,i+1]=U.dot(psi[:,i])
        n=np.sum(np.square(np.abs(psi[:,i+1])))
        psi[:,i+1]=psi[:,i+1]/np.sqrt(n)
    else:
        e2=np.random.rand()
        if e2 < p_down/p:
            psi[:,i+1]=[1,0]
        else:
            psi[:,i+1]=[0,1]
        
#inv=(np.square(abs(psi[1,:]))-np.square(abs(psi[0,:])))
norm=np.sum(np.square(np.abs(psi)),axis=0)

fig = plt.figure(num=1)

ax = fig.add_subplot(111)
ax.set_ylabel("Population",fontsize=14)
ax.set_xlabel("Time",fontsize=14)
ax.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(psi[0,:])), color='red')
ax.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(psi[1,:])), color='blue')
ax.plot(np.linspace(0,t,num=n_steps),norm,color='black', linestyle='dashed')
ax.legend(('Ground state', 'Excited state', 'Norm'))

fig.suptitle('Single quantum trajectory', fontsize=16)
