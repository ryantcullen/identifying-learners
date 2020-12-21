Author: Ryan T. Cullen

ABSTRACT:

The goal of this project is to attempt to develop a taxonomy for identifying whether or not an agent (or organism) is a reinforcement learner. A reinforcement learner is an agent that seeks to maximize its total reward by altering its action policy. However, simple observation of an agent acting to maximize its potential future reward is not enough to say that it has the ability to learn. The justification for this claim is that the population as a whole could be doing the learning over evolutionary timescales, as opposed to the individual agent learning over its lifetime. This paper makes an attempt at establishing a protocol for distinguishing between these two strategies. Reinforcement learning algorithms such as Policy Iteration and Q-learning are tested as a means of solving/learning to solve MDPs, alongside various detection methods to identify potential learners. Outlier detection proved to be sub-optimal overall under most conditions, but certain environment configurations yielded extremely successful trials. Other methods such as signal detection were explored as a means of improving the protocol.  Also discussed is the intriguing fact that an organism’s ability to learn had to have itself been learned by the population first. 




