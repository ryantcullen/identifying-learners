import numpy as np
import itertools
import pandas as pd
import sys
import timeit
from gridworld import GridworldEnv
from random import randint
from numpy import empty

env = GridworldEnv()
actionSteps = env.P

def epsilonGreedyPolicy(Q, epsilon, nA):
    
    def policyFunc(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        
        A[best_action] += (1.0 - epsilon)
        return A
    return policyFunc


def Qlearn(env, numEpisodes, discountFactor=0.9, alpha=0.5, epsilon=0.1):
    

    #initialize action-value and policy functions
    Q = np.random.uniform(size=[env.nS,env.nA])

    policy = epsilonGreedyPolicy(Q, epsilon, env.nA)
    
    for i_episode in range(numEpisodes):
        #track episodes in console becuase I am fancy and my programs are fancy
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, numEpisodes), end="")
            sys.stdout.flush()
        
        #reset agent to random starting position
        s = randint(0, env.nS - 1)
        
        #step through envirmonment
        while True:
            
            #take a step
            actionProbs = policy(s)
            a = np.random.choice(np.arange(len(actionProbs)), p=actionProbs)
            [(nextState, reward, done)] = actionSteps[s][a]
        
            #update Q
            bestNextAction = np.argmax(Q[nextState])    
            td_target = reward + discountFactor * Q[nextState][bestNextAction]
            td_delta = td_target - Q[s][a]
            Q[s][a] += alpha * td_delta
    
            if done:
                break
            
            s = nextState

    #extract policy from Q
    rows = env.shape[0]
    cols = env.shape[1]
    counter = 0
    extractedPolicy = np.zeros([rows,cols])
    for i in range(rows):
        for j in range (cols):
            extractedPolicy[i][j] = int(np.argmax(Q[counter]))
            counter += 1

    print("")
    print("")
    print("Action Value Function:")
    print(Q)

    print("")
    print("")
    print("Policy Map (0=up, 1=right, 2=down, 3=left):")
    print(extractedPolicy)
    print("")

    return Q, extractedPolicy

def functionWrapper(f, *args, **kwargs):
    def wrappedFunction():
        return f(*args, **kwargs)
    return wrappedFunction

wrappedQ = functionWrapper(Qlearn, env, 10000)
print("Execution Time:", timeit.timeit(wrappedQ, number = 1))

