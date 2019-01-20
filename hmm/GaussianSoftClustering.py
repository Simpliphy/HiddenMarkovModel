import numpy as np
from scipy.stats import multivariate_normal
import math
from numpy.linalg import det
from tqdm import tqdm

class GaussianSoftClustering(object):
    """
    Based on assignment from week 2 Bayesian method for machine learning of Coursera.
    """

    def __init__(self):

        self._nothing = None

    def gauss_den(self, x, mu, sigma, d):
        """
        computes the density given a mean vector mu (d,) cov matrix sigma (dxd) and X(Nxd)
        """
        x = np.matrix(x)
        mu = np.matrix(mu)
        sigma = np.matrix(sigma)
        return (1 / math.sqrt(math.pow(2 * math.pi, d) * det(sigma))) * np.diag(
            np.exp(-0.5 * (x - mu) * sigma.I * (x - mu).T))

    def E_step(self, X, pi, mu, sigma):
        """
        Performs E-step on GMM model
        Each input is numpy array:
        X: (N x d), data points
        pi: (C), mixture component weights
        mu: (C x d), mixture component means
        sigma: (C x d x d), mixture component covariance matrices

        Returns:
        gamma: (N x C), probabilities of clusters for objects
        """
        N = X.shape[0]  # number of objects
        C = pi.shape[0]  # number of clusters
        d = mu.shape[1]  # dimension of each object
        gamma = np.zeros((N, C))  # distribution q(T)


        # P(t|x)=p(x|t)p(t)/z
        # p(x|t)=N(mu,sigma)
        for t in range(C):
            gamma[:, t] = multivariate_normal.pdf(X, mean=mu[t, :], cov=sigma[t, ...]) * (pi[t])
            # gauss_den(X,mu[t,:],sigma[t,...],d)*(pi[t])

        gamma /= np.sum(gamma, 1).reshape(-1, 1)  # normalize by z

        return gamma

    def M_step(self, X, gamma):
        """
        Performs M-step on GMM model
        Each input is numpy array:
        X: (N x d), data points
        gamma: (N x C), distribution q(T)

        Returns:
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)
        """
        number_of_objects = X.shape[0]
        number_of_clusters = gamma.shape[1]
        number_of_features = X.shape[1]  # dimension of each object

        normalizer = np.sum(gamma, 0)  # (K,)
        # print normalizer.shape
        mu = np.dot(gamma.T, X) / normalizer.reshape(-1, 1)
        pi = normalizer / number_of_objects
        sigma = np.zeros((number_of_clusters, number_of_features, number_of_features))

        # for every k compute cov matrix
        for cluster_index in range(number_of_clusters):
            x_mu = X - mu[cluster_index]
            gamma_diag = np.diag(gamma[:, cluster_index])

            sigma_k = np.dot(np.dot(x_mu.T , gamma_diag) , x_mu)
            sigma[cluster_index, ...] = (sigma_k) / normalizer[cluster_index]

        return pi, mu, sigma


    def compute_vlb(self, X, pi, mu, sigma, gamma):
        """
        Each input is numpy array:
        X: (N x d), data points
        gamma: (N x C), distribution q(T)
        pi: (C)
        mu: (C x d)
        sigma: (C x d x d)

        Returns value of variational lower bound
        """
        N = X.shape[0]  # number of objects
        C = gamma.shape[1]  # number of clusters
        d = X.shape[1]  # dimension of each object

        ### YOUR CODE HERE

        loss = 0
        for k in range(C):
            dist = multivariate_normal(mu[k], sigma[k], allow_singular=True)
            for n in range(N):
                loss += gamma[n, k] * (np.log(pi[k] + 0.00001) + dist.logpdf(X[n, :]) - np.log(gamma[n, k] + 0.000001))

        loss = np.zeros(N)
        EPSILON = 1e-10
        for k in range(C):
            loss += gamma[:, k] * (np.log(pi[k]) + multivariate_normal.logpdf(X, mean=mu[k, :], cov=sigma[k, ...]) - \
                                   np.log(gamma[:, k]))
            # loss+=gamma[:,k]*(np.log(pi[k]*multivariate_normal.pdf(X, mean=mu[k,:], cov=sigma[k,...])+0)-np.log(gamma[:,k]))
            # loss+=gamma[:,k]*(np.log(pi[k]*gauss_den(X,mu[k,:],sigma[k,...],d)+EPSILON)-np.log(gamma[:,k]+EPSILON))

        return np.sum(loss)

    def train_EM(self, X, C, rtol=1e-3, max_iter=100, restarts=10):
        '''
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.

        X: (N, d), data points
        C: int, number of clusters
        '''
        N = X.shape[0]  # number of objects
        d = X.shape[1]  # dimension of each object

       # pi = 1 / float(C) * np.ones(C)
      #  mu = np.random.randn(C, d)
       # sigma = np.zeros((C, d, d))
       # sigma[...] = np.identity(d)

        best_loss = -1e7
        best_pi = None
        best_mu = None
        best_sigma = None
        best_gamma = None

        for _ in range(restarts):
            #print("restart")
           # try:

            pi = 1 / float(C) * np.ones(C)
            mu = np.random.randn(C, d)
            sigma = np.zeros((C, d, d))
            sigma[...] = np.identity(d)

            gamma = self.E_step(X, pi, mu, sigma)
            prev_loss = self.compute_vlb(X, pi, mu, sigma, gamma)

            for _ in tqdm(range(max_iter)):

                gamma = self.E_step(X, pi, mu, sigma)
                pi, mu, sigma = self.M_step(X, gamma)
                loss = self.compute_vlb(X, pi, mu, sigma, gamma)
                if loss / prev_loss < rtol:
                    break
                if loss > best_loss:
                    # print loss
                    best_loss = loss
                    best_pi = np.copy(pi)
                    best_mu = np.copy(mu)
                    best_sigma = np.copy(sigma)
                    best_gamma = np.copy(gamma)

                prev_loss = loss
                #print("step %d" % it)

           # except np.linalg.LinAlgError:
            #    print("Singular matrix: components collapsed")
            #    pass

        return best_loss, best_pi, best_mu, best_sigma, best_gamma

