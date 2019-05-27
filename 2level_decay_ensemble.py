import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import expm
%matplotlib notebook


n_steps=100000
n_samples=1000
delta_t=1e-4
t=delta_t*n_steps

omega_0=1
lam=0.6
g=0


H_eff=np.array([[-(omega_0)/2,lam],[lam,(omega_0-1j*g)/2]])
U=expm(-1j*H_eff*delta_t)
psi=np.zeros([2,n_steps,n_samples], dtype=complex)
psi[1,0,:]=np.ones(n_samples)
decayed_psi=np.zeros([2,n_samples], dtype=complex)
decayed_psi[0,:]=np.ones(n_samples)

#Run over time
for i in range(0,n_steps-1):
    
    progress = 100 * i / n_steps
    if progress % 5 == 0:
        print("Progress: " + str(progress) + "%")
    
    p=delta_t*g*np.square(np.abs(psi[1,i,:]))
    jump = np.random.rand(n_samples) < p
    psi[:,i+1,jump] = decayed_psi[:,jump]
    psi[:,i+1,np.logical_not(jump)] = U.dot(psi[:,i,np.logical_not(jump)])/np.sqrt(1-p[np.logical_not(jump)])

probs=np.mean(np.square(np.abs(psi)),axis=2)
phase=np.mean(np.angle(np.conj(psi[0,:,:])*psi[1,:,:]),axis=1)
#inv=(np.square(np.abs(state[1,:]))-np.square(np.abs(state[0,:])))
state=np.zeros([2,n_steps], dtype=complex)
state=np.sqrt(probs)
#state[1,:]=state[1,:]*np.exp(1j*phase)

fig = plt.figure(num=1)

ax = fig.add_subplot(111)
ax.set_ylabel("Population",fontsize=14)
ax.set_xlabel("Time",fontsize=14)
ax.plot(np.linspace(0,t,num=n_steps),probs[0,:],color='red')
ax.plot(np.linspace(0,t,num=n_steps),probs[1,:],color='blue')
ax.legend(('Ground state', 'Excited state'))

fig.suptitle('Ensemble trajectory', fontsize=16)
