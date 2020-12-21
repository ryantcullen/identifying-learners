import numpy as np
import sys
import gym.spaces
import timeit
if "../" not in sys.path:
  sys.path.append("../") 
from gridworld import GridworldEnv


environment = GridworldEnv()

def value_iteration(environment, discountFactor=0.9, minError=0.1):
    
    def lookahead(V, a, s):
        
        [(next_state, reward, done)] = environment.P[s][a]
        #Bellman eqn
        value = (reward + discountFactor * V[next_state])
        return value
    

    #inital value function and policy
    V = np.zeros(environment.nS)
    policy = np.zeros([environment.nS, environment.nA])


    while True:

        error = 0

        #loop over states
        for s in range(environment.nS):

            actions_values = np.zeros(environment.nA)
            
            for a in range(environment.nA):

                #apply Bellman eqn
                actions_values[a] = lookahead(V, a, s)


            #Update value function and error
            best_action_value = max(actions_values) 
            error = max(error, abs(best_action_value - V[s]))
            V[s] = best_action_value

            #Update policy function
            best_action = np.argmax(actions_values)
            policy[s] = np.eye(environment.nA)[best_action]


        #break if close enough to optimal value function
        if(error < minError):
            break
    
    print("")
    print("Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), environment.shape))
    print("")

    #print("Grid Value Function:")
    #print(V.reshape(environment.shape))
    #print("")
    
    return policy, V

def functionWrapper(f, *args):
    def wrappedFunction():
        return f(*args)
    return wrappedFunction

wrapped = functionWrapper(value_iteration, environment)
print("Execution Time:", timeit.timeit(wrapped, number = 1))