# yonghong HUANG
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gamma


class SVI:
    def _init_(self, N, limit = 0.005, a0=0, b0=0, mu0=0, lambda0=0):
        self.a0 = a0
        self.b0 = b0
        self.mu0 = mu0
        self.lambda0 = lambda0
        self.N = N
        self.datalist = np.random.normal(0, 1, self.N)
        self.limit = limit

        self.lambdaN = False
        self.aN = False
        self.bN = False
        self.muN = False

    def calbN(self):
        Emusquare = self.muN ** 2 + 1 / self.lambdaN
        self.bN = self.b0 + sum(
            [self.datalist[i] ** 2 + Emusquare - 2 * self.muN * self.datalist[i] for i in range(self.N)]) / 2 + \
                  self.lambda0 * (Emusquare + self.mu0 ** 2 - 2 * self.mu0 * self.muN)

    def VIalgorithm(self):
        self.aN = self.a0 + 0.5*self.N
        self.muN = (self.lambda0 * self.mu0 + np.sum(self.datalist)) / (self.lambda0 + self.N)

        self.lambdaN = 1
        Emusquare = self.muN ** 2 + 1 / self.lambdaN

        self.bN = self.b0+sum([self.datalist[i]**2+Emusquare-2*self.muN*self.datalist[i] for i in range(self.N)])/2 + \
                  self.lambda0*(Emusquare+self.mu0**2-2*self.mu0*self.muN)
        self.lambdaN = (self.lambda0+self.N) * (self.aN/self.bN)

        ite = 10
        while (ite > 0):
            ite -= 1
            self.calbN()
            lambdaN = (self.lambda0 + self.N) * self.aN / self.bN
            if np.abs(self.lambdaN - lambdaN) / self.lambdaN < self.limit:
                break
            self.lambdaN = lambdaN

    def postvi(self,mu,tao):
        return multivariate_normal.pdf(mu, mean=self.muN, cov=self.lambdaN)*gamma.pdf(tao,self.aN,scale=1/self.bN)


