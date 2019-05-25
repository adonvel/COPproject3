import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import expm
%matplotlib notebook


n_steps=3000
delta_t=1e-3
t=delta_t*n_steps
n_samples=1000

omega_e=1
omega_g=-1

lam12=3
lam13=3
lam23=3
g_21=1.0
g_31=1.0
g_32=1.0

H_eff=np.array([[-omega_g,lam12,lam13],[lam12,-1j*g_21,lam23],[lam13,lam23,omega_e-1j*(g_31+g_32)]])
#U=np.array([[1,0,0],[0,1,0],[0,0,1]])-1j*H_eff*delta_t
U=expm(-1j*H_eff*delta_t)

psi=np.zeros([3,n_steps], dtype=complex)
psi[:,0]=[0,0,1]

psi_save=np.zeros([3,n_steps,n_samples], dtype=complex)
psi_save[2,0,:]=np.ones(n_samples)


#Run over samples

for j in range(0,n_samples):
    
    progress = 100 * j / n_samples
    if progress % 5 == 0:
        print("Progress: " + str(progress) + "%")
    
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

fig = plt.figure(num=1)

ax = fig.add_subplot(111)
ax.set_ylabel("Population",fontsize=14)
ax.set_xlabel("Time",fontsize=14)
ax.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(state[0,:])), color='red')
ax.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(state[1,:])), color='green')
ax.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(state[2,:])), color='blue')
ax.plot(np.linspace(0,t,num=n_steps),norm,color='black', linestyle='dashed')
ax.legend(('Ground state', 'Intermediate state', 'Excited state', 'Norm'))

fig.suptitle('Ensemble trajectory', fontsize=16)
