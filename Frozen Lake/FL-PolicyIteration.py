import gym
import numpy as np
import math
import timeit
import numpy.linalg as LA

env = gym.make('FrozenLake-v0')

S = env.observation_space.n
A = env.action_space.n

def MDPparams(S, A):

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
            r[j][i] = env.P[j][i][0][2]

    return r, trans_p

def solveMDP(r,trans_p):
    # start with random initial policy
    # only have to search over deterministic policies.

    gamma = 0.99
    pi_old = ['0']*S
    dpi = True

    while dpi:
        
        # calculate V of pi
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

    
    reshapedPolicy = ReshapePolicy(S, pi_old)
    print("Policy Map (0=left, 1=down, 2=right, 3=up):")
    print(reshapedPolicy)
    print("")
    print("Environment:")
    env.render()

    return reshapedPolicy, avg_r

#turn policy function from a list into a matrix
def ReshapePolicy(S,pi):
    rows = int(math.sqrt(S))
    cols = int(math.sqrt(S))
    reshapedPolicy = np.zeros([rows,cols])
    counter = 0
    for i in range(rows):
        for j in range (cols):
            reshapedPolicy[i][j] = int(pi[counter])
            counter += 1
    return reshapedPolicy

#initialize reward and transition matrices
rewards, transProbs = MDPparams(env.nS, env.nA)

#wrap function so we cant time it
def functionWrapper(f, *args, **kwargs):
    def wrappedFunction():
        return f(*args, **kwargs)
    return wrappedFunction

wrapped = functionWrapper(solveMDP, rewards, transProbs)
print("Execution Time:", timeit.timeit(wrapped, number = 1))
