import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.2
        self.gamma = 1.0
        self.epsilon = 1
        self.decay = 0.9
        self.min_epsilon = 0.05

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        action = np.argmax(self.Q[state])

        return action


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        #using expected sarsa
        if(not done):
            policy = np.ones(self.nA) * self.epsilon / self.nA
            policy[np.argmax(self.Q[next_state])] = 1 - self.epsilon + self.epsilon/self.nA

            self.Q[state][action] = self.Q[state][action] + self.alpha * ( reward + self.gamma * np.dot(self.Q[next_state],policy) - self.Q[state][action] )
        else:
            self.Q[state][action] = self.Q[state][action] + self.alpha * ( reward - self.Q[state][action] )

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

        
        #self.Q[state][action] += 1
