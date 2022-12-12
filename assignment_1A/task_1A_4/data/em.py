# yonghong HUANG
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, multivariate_normal
import math

class SEEM:
    def __init__(self, k, iter, limit=0.001):
        self.limit = limit
        self.iter = iter
        self.X = self.getX()
        self.S = self.getS()
        self.number = len(self.X)
        self.k = k
        self.r = np.zeros((self.number, self.k))
        self.pi = []
        self.mu = [] #lambda
        self.cov = []
        self.miu = np.zeros((self.k, 2))
        self.likelihood = 0


    def getX(self):
        X = np.loadtxt('X.txt')
        return X

    def getS(self):
        S = np.loadtxt('S.txt')
        return S

    def initia(self):
        np.random.seed(0)
        self.pi = np.full(self.k, 1.0 / self.k)
        self.mu = np.mean(self.S) * np.ones(self.k)
        self.miu = np.mean(self.X) * np.ones(self.k)
        self.cov = np.ones(self.k)


    def E_part(self):
        r = []
        for i in range(self.number):
            s = 0
            for j in range(self.k):
                self.r[i][j] = self.pi[j] * poisson.pmf(self.S[i],self.mu[j])*\
                               multivariate_normal.pdf(self.X[i],mean=self.miu[j],cov=self.cov[j])

                s += poisson.pmf(self.S[i],self.mu[j])*\
                               multivariate_normal.pdf(self.X[i],mean=self.miu[j],cov=self.cov[j])
            r.append(s)
        for i in range(self.number):
            for j in range(self.k):
                self.r[i][j] /= r[i]


    def M_part(self):
        for k in range(self.k):
            n = np.sum(self.r[:,k])
            self.miu[k]= np.dot(np.array(self.r[:, k]), self.X) / n
            self.mu[k] = np.dot(self.S, np.array(self.r[:, k])) / n
            self.pi[k] = n / self.number
            for i in range(self.number):
                self.cov[k] += self.r[i][k]*(np.matrix(self.X[i]-self.miu[k])*np.matrix(self.X[i]-self.miu[k]).T)
            self.cov[k] /= n

    def loglike(self):
        for i in range(self.number):
            for j in range(self.k):
                self.r[i][j] = self.pi[j] * poisson.pmf(self.S[i], self.mu[j]) \
                               * multivariate_normal.pdf(self.X[i], self.miu[j], self.cov[j])
        likelihood = np.sum(np.log(np.sum(self.r, axis=1)))
        if np.abs(self.likelihood-likelihood) < self.limit:
            return False
        self.likelihood = likelihood
        return True

    def iteration(self):
        i = 0
        self.initia()
        while(i<self.iter):
            self.E_part()
            self.M_part()
            if(self.loglike()== False):
                break
            i += 1


f = SEEM(2,50)
f.iteration()




