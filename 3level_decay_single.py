import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import expm
%matplotlib notebook


n_steps=9000
delta_t=1e-3
t=delta_t*n_steps

omega_e=1
omega_g=-1
lam12=1   #1
lam13=2  #20
lam23=2   #0
g_21=0.1 #0.01
g_31=2    #2
g_32=0.5   #0


H_eff=np.array([[-omega_g,lam12,lam13],[lam12,-1j*g_21,lam23],[lam13,lam23,omega_e-1j*(g_31+g_32)]])
#U=np.array([[1,0,0],[0,1,0],[0,0,1]])-1j*H_eff*delta_t
U=expm(-1j*H_eff*delta_t)
psi=np.zeros([3,n_steps], dtype=complex)
psi[:,0]=[0,0,1]
signal=np.zeros(n_steps)

#Run over time
for i in range(0,n_steps-1):
    p21=delta_t*g_21*np.square(abs(psi[1,i]))
    p31=delta_t*g_31*np.square(abs(psi[2,i]))
    p32=delta_t*g_32*np.square(abs(psi[2,i]))
    p=p21+p31+p32
    e=np.random.rand()
    if e > p:
        psi[:,i+1]=U.dot(psi[:,i])
        n=np.sum(np.square(np.abs(psi[:,i+1])))
        psi[:,i+1]=psi[:,i+1]/np.sqrt(n)
    else:
        e2=np.random.rand()
        if e2 < (p21+p31)/p:
            psi[:,i+1]=[1,0,0]
            if e2>p21/p:
                signal[i]=1
        else:
            psi[:,i+1]=[0,1,0]
            
norm=np.sum(np.square(np.abs(psi)),axis=0)

fig = plt.figure(num=1)

ax = fig.add_subplot(111)
ax.set_ylabel("Population",fontsize=14)
ax.set_xlabel("Time",fontsize=14)
ax.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(psi[0,:])), color='red')
ax.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(psi[1,:])), color='green')
ax.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(psi[2,:])), color='blue')
ax.plot(np.linspace(0,t,num=n_steps),norm,color='black', linestyle='dashed')
ax.legend(('Ground state', 'Intermediate state', 'Excited state', 'Norm'))
fig.suptitle('Single quantum trajectory', fontsize=16)

fig_sig=plt.figure(num=2)
ax2 = fig_sig.add_subplot(111)
ax2.plot(np.linspace(0,t,num=n_steps),signal)
