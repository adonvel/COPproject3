import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib notebook


n_steps=100000
n_samples=1000
delta_t=1e-4
t=delta_t*n_steps

omega_0=2
beta=1
lam=2
g_down=0.5
g_up=0
#g_up=np.exp(-beta*omega_0)*g_down


H_eff=np.array([[-(omega_0+1j*g_up)/2,lam],[lam,(omega_0-1j*g_down)/2]])
U=np.array([[1,0],[0,1]])-1j*H_eff*delta_t
psi=np.zeros([2,n_steps,n_samples], dtype=complex)
psi[1,0,:]=np.ones(n_samples)
decayed_psi=np.zeros([2,n_samples], dtype=complex) #I don't know if this is the best way todo this
decayed_psi[0,:]=np.ones(n_samples)

#Run over time
for i in range(0,n_steps-1):
    
    progress = 100 * i / n_steps
    if progress % 5 == 0:
        print("Progress: " + str(progress) + "%")
    
    p=delta_t*g_down*np.square(np.abs(psi[1,i,:]))
    jump = np.random.rand(n_samples) < p
    psi[:,i+1,jump] = decayed_psi[:,jump]
    psi[:,i+1,np.logical_not(jump)] = U.dot(psi[:,i,np.logical_not(jump)])/np.sqrt(1-p[np.logical_not(jump)])

state=np.mean(psi,axis=2)
norm=np.sum(np.square(np.abs(state)),axis=0)
inv=(np.square(np.abs(state[1,:]))-np.square(np.abs(state[0,:])))

fig = plt.figure(num=1)

ax = fig.add_subplot(111)
ax.set_ylabel("Population",fontsize=14)
ax.set_xlabel("Time",fontsize=14)
ax.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(state[0,:])), color='red')
ax.plot(np.linspace(0,t,num=n_steps),np.square(np.abs(state[1,:])), color='blue')
ax.plot(np.linspace(0,t,num=n_steps),norm,color='black', linestyle='dashed')
ax.legend(('Ground state', 'Excited state', 'Norm'))

fig.suptitle('Ensemble trajectory', fontsize=16)
