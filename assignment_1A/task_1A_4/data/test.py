# yonghong HUANG
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
import math

class SEEM:
    def __init__(self, k, iter, limit=0.000001):
        self.limit = limit
        self.iter = iter
        self.X0 = []
        self.X1 = []
        self.S = []
        self.number = 0
        self.k = k
        self.r = []
        self.pi = []
        self.lamda= [] #lambda
        self.cov0 = []
        self.cov1 = []
        self.miu0 = np.zeros(self.k)
        self.miu1 = np.zeros(self.k)
        self.likelihood = 0


    def getX(self):
        X0 = []
        X1 = []
        R = np.loadtxt('X.txt')
        b = np.hsplit(R, 2)
        i = 0
        for item in b[0]:
            X0.append(item[0])
            i += 1
        i = 0
        for item in b[1]:
            X1.append(item[0])
            i += 1
        self.X1 = X1
        self.X0 = X0

    def getS(self):
        self.S = np.loadtxt('S.txt')


    def initia(self):
        self.getX()
        self.getS()
        self.number = len(self.S)
        np.random.seed(0)
        self.pi = np.full(self.k, 1.0 / self.k)
        self.lamda = np.mean(self.S) * np.ones(self.k)

        self.cov0 = np.ones(self.k)
        self.cov1 = np.ones(self.k)
        self.r = [[0]*self.k]*self.number


        self.miu0 = np.mean(self.X0) * np.ones(self.k)
        self.miu1 = np.mean(self.X1) * np.ones(self.k)

    def E_part(self):
        r = []
        for i in range(self.number):
            s = 0
            for j in range(self.k):
                self.r[i][j] = self.pi[j] * poisson.pmf(self.S[i],self.lamda[j])*\
                               norm.pdf(self.X0[i],loc=self.miu0[j],scale=self.cov0[j])\
                               *norm.pdf(self.X1[i],loc=self.miu1[j],scale=self.cov1[j])

                s += self.r[i][j]
            r.append(s)
        self.r = np.array(self.r)
        r = np.array(r)
        self.r = self.r.astype(np.float64)
        r = r.astype(np.float64)
        for i in range(self.number):
            for j in range(self.k):
                self.r[i][j] /= r[i]

    def M_part(self):

        for k in range(self.k):

            # print(np.array(self.X0).shape)
            # print(np.array(self.r[:,k]).shape)
            n = np.sum(self.r[:, k])
            self.miu0[k] = np.dot(self.r[:, k], self.X0) / n
            self.miu1[k] = np.dot(self.r[:, k], self.X1) / n
            self.lamda[k] = np.dot(self.S, self.r[:, k]) / n
            self.pi[k] = n / self.number
            for i in range(self.number):
                self.cov0[k] += self.r[i][k] * ((self.X0[i]-self.miu0[k])**2)
                self.cov1[k] += self.r[i][k] * ((self.X0[i]-self.miu0[k])**2)
            self.cov0[k] /= n
            self.cov1[k] /= n

    def loglike(self):
        p = np.zeros((self.number, self.k))
        for i in range(self.number):
            for j in range(self.k):
                p[i][j] = self.pi[j] * poisson.pmf(self.S[i], self.lamda[j])\
                          * norm.pdf(self.X0[i], loc=self.miu0[j], scale=self.cov0[j])\
                          * norm.pdf(self.X1[i], loc=self.miu1[j], scale=self.cov1[j])
        likelihood = np.sum(np.log(np.sum(p, axis=1)))
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

k = 3
f = SEEM(k,20)
f.iteration()

strthlist = np.zeros(f.number)
locationlist = np.zeros((f.number, 2))
catlist = np.zeros(f.number)

#creat datalist

for i in range(f.number):
    j = np.random.choice(k, p=f.pi)
    x0 = np.random.normal(loc=f.miu0[j], scale=f.cov0[j])
    x1 = np.random.normal(loc=f.miu1[j], scale=f.cov1[j])
    locationlist[i][0] = x0
    locationlist[i][1] = x1
    strthlist[i] = np.random.poisson(f.lamda[j])
    catlist[i] = j

#plot
plt.figure(1)
plt.scatter(f.X0, f.X1, s=f.S)
plt.show()

plt.figure(2)
plt.scatter(locationlist[:,0],locationlist[:,1], strthlist)
plt.show()





