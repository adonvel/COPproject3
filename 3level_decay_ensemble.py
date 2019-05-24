import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import expm
%matplotlib notebook


n_steps=3000
delta_t=1e-3
t=delta_t*n_steps

omega_e=1
omega_g=-1
lam12=3
lam13=4
g_21=0.3
g_31=0.3
g_32=0.9

H_eff=np.array([[-omega_g,lam12,lam13],[lam12,-1j*g_21,0],[lam13,0,omega_e-1j*(g_31+g_32)]])
#U=np.array([[1,0,0],[0,1,0],[0,0,1]])-1j*H_eff*delta_t
U=expm(-1j*H_eff*delta_t)

psi=np.zeros([3,n_steps], dtype=complex)
psi[:,0]=[0,0,1]

psi_save=np.zeros([3,n_steps,n_samples], dtype=complex)
psi_save[2,0,:]=np.ones(n_samples)


#Run over samples

for j in range(0,n_samples):
    
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
            else:
                psi[:,i+1]=[0,1,0]
    psi_save[:,:,j]=psi
    
state=np.mean(psi_save,axis=2)   
norm=np.sum(np.square(np.abs(state)),axis=0)

plt.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(state[0,:])))
plt.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(state[1,:])))
plt.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(state[2,:])))
plt.plot(np.linspace(0,t,num=n_steps),norm)
