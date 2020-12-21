import numpy as np
import gym.spaces
import timeit
import numpy.linalg as LA
from gridworld import GridworldEnv

env = GridworldEnv()

def MDPparams(S, A):
    trans_p = {}
    r = np.random.uniform(size=[env.nS,env.nA])

    for i in range(A):
        M = np.zeros([S,S])
        for j in range(S):
            M[:,j] = np.random.dirichlet(alpha=np.ones(S))
            trans_p[str(i)] = M
    return r, trans_p

def solveMDP(r,trans_p):
    # start with random initial policy
    # only have to search over deterministic policies.
    S = env.nS
    A = env.nA
    gamma = 0.99
    pi_old = ['0']*S
    dpi = True
    print(trans_p)
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

rewards, transProbs = MDPparams(env.nS, env.nA)

pi, avR = solveMDP(rewards, transProbs)

print("Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(pi, env.shape))