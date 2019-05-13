import numpy as np
from matplotlib import pyplot as plt

class HMM(object):
    # Construct an HMM with the following set of variables
    #    numStates:          The size of the state space
    #    numOutputs:         The size of the output space
    #    trainStates[i][j]:  The jth element of the ith state sequence
    #    trainOutputs[i][j]: Similarly, for output
    def __init__(self, numStates, numOutputs, states):
        self.numStates = numStates
        self.numOutputs = numOutputs
        self.states = states

        # Numbering the rooms column-wise from lower-left,
        # i.e., (0,0) = 0; (0,1) = 1; 0... (1,0) = 4; 0...

        # Numbering observations as r = 0; g = 1; b = 2; y = 3

        # Self-transitions
        self.T = np.zeros((self.numStates, self.numStates))

        for i in range(self.numStates):
            self.T[i,i] = 0.2
        # Black rooms
        self.T[0,0] = 1
        self.T[3,3] = 1
        self.T[5,5] = 1
        self.T[14,14] = 1
        self.T[2,1] = 0.8
        self.T[1,2] = 0.4
        self.T[6,2] = 0.4
        self.T[8,4] = 0.8
        self.T[2,6] = 0.8 / 3
        self.T[7,6] = 0.8 / 3
        self.T[10,6] = 0.8 / 3
        self.T[6,7] = 0.4
        self.T[11,7] = 0.4
        self.T[4,8] = 0.8 / 3
        self.T[9,8] = 0.8 / 3
        self.T[12,8] = 0.8 / 3
        self.T[8,9] = 0.8 / 3
        self.T[10,9] = 0.8 / 3
        self.T[13,9] = 0.8 / 3
        self.T[9,10] = 0.8 / 3
        self.T[11,10] = 0.8 / 3
        self.T[6,10] = 0.8 / 3
        self.T[7,11] = 0.8 / 3
        self.T[10,11] = 0.8 / 3
        self.T[15,11] = 0.8 / 3
        self.T[8,12] = 0.4
        self.T[13,12] = 0.4
        self.T[12,13] = 0.4
        self.T[9,13] = 0.4
        self.T[11,15] = 0.8

        # Emission likelihoods
        self.M = np.ones((4,16)) * 0.1
        self.M[:,[0,3,5,14]] = 0.25
        self.M[0,1] = 0.7
        self.M[1,2] = 0.7
        self.M[1,4] = 0.7
        self.M[2,6] = 0.7
        self.M[0,7] = 0.7
        self.M[3,8] = 0.7
        self.M[1,9] = 0.7
        self.M[0,10] = 0.7
        self.M[3,11] = 0.7
        self.M[2,12] = 0.7
        self.M[3,13] = 0.7
        self.M[2,15] = 0.7

        # Prior probabilities
        self.pi = np.ones((16,1)) / 12
        self.pi[[0,3,5,14]] = 0



    # Estimate the transition and observation likelihoods and the
    # prior over the initial state based upon training data
    def train(self, obs):

        n = len(obs)
        obs_dist = np.expand_dims(self.obsIndicator(obs), -1)
        eps = 1.0e-8
        logp = [-100001, -100000]
        while (logp[-1] - logp[-2]) > 0.01:
            allAlphas = []
            allBetas = []
            allGammas = np.zeros((len(obs), len(obs[0]), self.numStates))
            allXis = np.zeros((len(obs),len(obs[0])-1, self.numStates, self.numStates))
            logz = 0;
            for i in range(len(obs)):
                alphas, C = self.forward(obs[i])
                betas = self.backward(obs[i], C)
                gammas = self.computeGammas(alphas, betas, obs[i])
                xis = self.computeXis(alphas, betas, obs[i])

                # In computing the measurement likelihood from the last alpha,
                # we need to account for the compounding effect of the
                # normalization terms.
                logz = logz + np.log(np.sum(alphas[-1])) + np.log(C).sum()

                allAlphas.append(alphas)
                allBetas.append(betas)
                allGammas[i] = gammas
                allXis[i] = xis

            # Compute the current estimate of the prior distribution
            # pi' = gamma_0, where we average over all samples
            pi = np.mean(allGammas[:,0:1],axis=0).T

            # Compute the estimate of the transition function based
            # upon xi and gamma for all samples. We include an epsilon
            # to avoid divide-by-zero
            T = np.zeros((16,16))
            # for t in range(200):
            #     denom = np.expand_dims(np.sum(allGammas[t,:],axis=0), 2)
            #     T = T + np.sum(allXis[t,:], axis=0)/(denom+eps)
            #
            # T = T/(np.sum(T,axis=0) + eps)
            denom = np.expand_dims (np.sum(allGammas[:,:-1], axis=(0,1)), 0)
            T = np.sum(allXis, axis=(0,1)) / (denom + eps)

            denom = np.sum(allGammas, axis=(0,1))
            M = np.sum(np.expand_dims(allGammas, 2) * obs_dist, axis=(0,1)) / (denom + eps)

            self.pi = pi
            self.T = T
            self.M = M

            logp.append(logz/n)
            print('logp: {}'.format(logp[-1]))

        self.plot(logp[2:])


    def viterbi(self, obs):

        # Use log for numerical precision
        logM = np.log(self.M)
        logT = np.log(self.T)

        AllXS = []

        for i in range(len(obs)):
            z = obs[i]
            delta = np.expand_dims(logM[z[0],:], axis=1) + np.log(self.pi)
            deltas = [delta]
            pres = [None]

            # Compute delta and pre for each time step
            for t in range(1,len(z)):
                trans = logT + deltas[-1].T
                delta = np.expand_dims(logM[z[t],:], axis=1) + np.expand_dims(np.amax(trans, axis=1), axis=1)
                deltas.append(delta)
                pre = np.argmax(trans, axis=1)
                pres.append(pre)

            # Now, backtrack to get the MAP sequence
            XS = [np.argmax(deltas[-1])]
            for t in range(len(z))[:0:-1]:
                XS = [pres[t][XS[0]]] + XS

            AllXS.append(XS)

        return AllXS


    def forward(self, z):
        alpha = self.M[z[0:1]].T * self.pi
        C = [np.sum(alpha)]
        alphas = [alpha / C[-1]]
        for t in range(1, len(z)):
            alpha = self.M[z[t:t+1]].T * self.T.dot(alphas[-1])
            C.append(np.sum(alpha))
            alphas.append(alpha / C[-1])
        return alphas, C


    def backward(self, z, C):
        betas = [np.ones((self.numStates,1)) / C[-1]]
        for t in range(len(z)-1)[::-1]:
            beta = self.M.dot(self.T * betas[0])[z[t+1:t+2]].T
            betas = [beta / C[t]] + betas
        return betas


    def computeGammas(self, alphas, betas, z):
        gammas = np.zeros((len(z), self.numStates))
        for t in range(len(z)):
            gamma = (alphas[t] * betas[t])[:,0]
            gammas[t] = gamma / np.sum(gamma)
        return gammas


    def computeXis(self, alphas, betas, z):
        xis = np.zeros((len(z)-1, self.numStates, self.numStates))
        eps = 1.0e-8
        for t in range(len(z)-1):
            alpha = alphas[t]
            beta = betas[t+1]
            #xi = alpha.T * self.T * self.M[z[t+1:t+2]].T * beta
            xi = self.M[z[t+1:t+2]].T * (beta * self.T * alpha.T)
            xis[t] = xi
        return xis


    def obsIndicator(self, obs):
        n = len(obs)
        Tmax = len(obs[0])
        indicator = np.zeros((n, Tmax, self.numOutputs))
        for i in range(n):
            ob = np.zeros((Tmax, self.numOutputs))
            for j in range(Tmax):
                ob[j,obs[i][j]] += 1
            indicator[i] = ob
        return indicator



    # Returns the log probability associated with a transition from
    # the dummy start state to the given state according to this HMM
    def getLogStartProb (self, state):

        # Your code goes here
        return np.log(self.pi[state])

    # Returns the log probability associated with a transition from
    # fromState to toState according to this HMM
    def getLogTransProb (self, fromState, toState):

        # Your code goes here
        return np.log(self.T[toState, fromState])

    # Returns the log probability of state state emitting output
    # output
    def getLogOutputProb (self, state, output):

        # Your code goes here
        return np.log(self.M[output, state])

    def plot(self, logp):
        plt.plot(logp, 'k', linewidth=3)
        plt.title('Data Log Likelihood')
        plt.xlabel('Iterations')
        plt.ylabel('Log Likelihood')
        plt.savefig('log-likelihood.png')
        plt.show()
