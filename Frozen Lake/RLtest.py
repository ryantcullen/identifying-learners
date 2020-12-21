import gym
import numpy as np
import pylab as pl
import math
import timeit
import numpy.linalg as LA
from scipy.stats import norm

# Authors: Dr. Sarah Marzen, Ryan Cullen
 
def MDPparams():
    env = gym.make('FrozenLake-v0')

    S = env.observation_space.n
    A = env.action_space.n
    trans_p = {}
    r = np.zeros([S,A])

    for i in range (A):
        M = np.zeros([S,S])
        for j in range(S):
            #set up transition matrix
            lis = [0]*S
            lis[env.P[j][i][0][1]] = 1
            M[:,j] = lis
            trans_p[str(i)] = M
            #set up reward matrix
            r[j][i] = np.random.uniform(-10, 10)
            #r[j][i] = env.P[j][i][0][2]
    return r, trans_p


# METHOD: Policy Iteration
def solveMDP(r,trans_p):
    # start with random initial policy
    # only have to search over deterministic policies.

    S, A = r.shape
    gamma = 0.99
    pi_old = ['0']*S
    dpi = True
    while dpi:
        
        # calculate V of pi_0
        B = np.zeros([S,S])
        for i in range(S):
            B[i,:] = trans_p[pi_old[i]][:,i]
        r2 = np.zeros(S)
        for i in range(S):
            r2[i] = r[i,int(pi_old[i])]
        V = np.dot(LA.inv(np.identity(S)-gamma*B),r2)

        # calculate Q
        Q = np.zeros([S,A])
        for i in range(A):
            Q[:,i] = r[:,i]+gamma*np.dot(trans_p[str(i)].T,V)

        # improve the policy
        dpi = False
        for i in range(S):
            new = str(np.argmax(Q[i,:]))
            if pi_old[i]!=new:
                dpi = True
            pi_old[i] = str(np.argmax(Q[i,:]))
        
            
    # calculate average reward
    M = np.zeros([S,S])
    for j in range(S):
        M[:,j] = trans_p[pi_old[j]][:,j]
    w, v = LA.eig(M)
    ind = np.argmin(np.abs(w-1))
    stationary_dist = v[:,ind]; stationary_dist /= np.sum(stationary_dist)
    avg_r = np.dot(V,stationary_dist)*(1-gamma)

    return pi_old, avg_r


def run_Envt(r,trans_p,pi,train_time,algorithm='None'):
    S, A = r.shape
    if algorithm=='None':
        # run it for all of train_time
        s = np.random.choice(S)
        a = np.random.choice(A, p=pi[s,:])
        R = r[s,a]
        for i in range(train_time-1):
            # sample to get s
            s = np.random.choice(S,p=trans_p[str(a)][:,s])
            a = np.random.choice(A, p=pi[s,:])
            R += r[s,a]
        R /= train_time
    elif algorithm=='Q-learning':

        # Run with Q learning
        s0 = np.random.choice(S)
        gamma = 0.99
        a0 = '0'
        R = r[s0,int(a0)]

        # Initialize Action-Policy function
        Q_Q = 10*np.mean(r)*np.ones(shape=[S,A])
        alpha = 0.1

        for i in range(train_time-1):

            # Sample the environment
            s1 = np.random.choice(a=S,p=trans_p[a0][:,s0])

            # Update Action-Policy Function
            dQ_Q = r[s0,int(a0)]+gamma*np.max(Q_Q[s1,:])-Q_Q[s0,int(a0)]
            Q_Q[s0,int(a0)] += alpha*dQ_Q/(i+1)

            # Explore vs. exploit
            u = np.random.uniform()
            epsilon = 0.1
            if u<1-epsilon:
                a0 = str(np.argmax(Q_Q[s1,:]))
            else:
                a0 = str(np.random.choice(A))
            
            # Update R
            R += r[s0,int(a0)]
            a0 = str(a0)
            s0 = s1
        R /= train_time
    else:
        R = 0
    return R

def ROC_curve(train_time):
    # have them acclimate to environment
    r1, trans_p1 = MDPparams()
    pi_old, r_avg = solveMDP(r1,trans_p1)
    S, A = r1.shape
    # turn the pi from solveMDP into a matrix of S by A
    pi = np.zeros([S,A])
    for i in range(S):
        pi[i,int(pi_old[i])] = 1
    # generate new environment
    r2, trans_p2 = MDPparams()
    # figure out the distribution for unchanging pi; this gives the null hypothesis
    r_null = []
    for i in range(1000):
        r_null.append(run_Envt(r2,trans_p2,pi,train_time,algorithm='None'))
    r_null = np.asarray(r_null)
    mean_r_null = np.mean(r_null)
    std_r_null = np.std(r_null)
    p_null = np.exp(-(r_null-mean_r_null)**2/2/std_r_null**2)/np.sqrt(2*np.pi*std_r_null**2)
    # then run for all the algorithms
    r_Qlearning = []
    for i in range(1000):
        r_Qlearning.append(run_Envt(r2,trans_p2,pi,train_time,algorithm='Q-learning'))
    r_Qlearning = np.asarray(r_Qlearning)
    mean_r_Qlearning = np.mean(r_Qlearning)
    p_Qlearning = np.exp(-(r_Qlearning-mean_r_null)**2/2/std_r_null**2)/np.sqrt(2*np.pi*std_r_null**2)
    # calculate FPR and TPR for each algorithm separately
    # p(r) = np.exp(-(r-meanr)^2/2/var_r)/np.sqrt(2*pi*var_r)
    thresholds = np.linspace(np.min(r_null) ,np.max(r_Qlearning),1000)
    FPR = np.zeros(shape=[1,len(thresholds)])
    TPR = np.zeros(shape=[1,len(thresholds)])
    
    # look for false positive rate for Q learning methods
    for i in range(len(thresholds)):
        FPR[0,i] = np.sum((r_null>thresholds[i]))/len(r_null)
        TPR[0,i] = np.sum((r_Qlearning>thresholds[i]))/len(r_Qlearning)
    
    #Calculate Area Under Curve
    AUC = 0
    for i in range(len(thresholds) - 1):
        AUC -= TPR[0,i] * (FPR[0,i+1]-FPR[0,i])

    return FPR, TPR, r_Qlearning, r_null, AUC


FPR, TPR, rQ, rNull, AUC = ROC_curve(50)

print(AUC)


mean,std = norm.fit(rQ)
pl1 = pl.hist(rQ, bins=20, density=True)
xmin, xmax = pl.xlim()
x = np.linspace(xmin, xmax, 100)
y = norm.pdf(x, mean, std)
pl.plot(x, y)

mean,std = norm.fit(rNull)
pl2 = pl.hist(rNull, bins=20, density=True)
xmin, xmax = pl.xlim()
x = np.linspace(xmin, xmax, 100)
y = norm.pdf(x, mean, std)
pl.plot(x, y)

pl.show() 


pl.plot(FPR.T,TPR.T)
pl.xlabel('FPR')
pl.ylabel('TPR')
pl.show()