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

        pass

    def E_step(self, observations, pi, mu, sigma):
        """
        Performs E-step on GMM model
        # P(t|x)=p(x|t)p(t)/z
        # p(x|t)=N(mu,sigma)

        Returns:
        gamma: (N x C), probabilities of clusters for objects
        """
        assert isinstance(observations, np.ndarray)

        number_of_observations = observations.shape[0]
        number_of_clusters = pi.shape[0]
        gamma = np.zeros((number_of_observations, number_of_clusters))  # distribution q(T)

        for cluster_index in range(number_of_clusters):

            multivariate_normal_pdf = multivariate_normal.pdf(observations,
                                                              mean=mu[cluster_index, :],
                                                              cov=sigma[cluster_index, ...])

            gamma[:, cluster_index] = multivariate_normal_pdf * (pi[cluster_index])

        gamma /= np.sum(gamma, 1).reshape(-1, 1)  # normalize by z

        return gamma

    def M_step(self, observations, states_distributions):
        """
        Performs M-step on GMM model
        """

        assert isinstance(observations, np.ndarray)
        assert isinstance(states_distributions, np.ndarray)

        number_of_objects = observations.shape[0]
        number_of_clusters = states_distributions.shape[1]
        number_of_features = observations.shape[1]  # dimension of each object

        normalization_constants = np.sum(states_distributions, 0)  # (K,)

        mu = np.dot(states_distributions.T, observations) / normalization_constants.reshape(-1, 1)
        pi = normalization_constants / number_of_objects
        sigma = np.zeros((number_of_clusters, number_of_features, number_of_features))

        for cluster_index in range(number_of_clusters):

            x_mu = observations - mu[cluster_index]
            gamma_diag = np.diag(states_distributions[:, cluster_index])

            sigma_k = np.dot(np.dot(x_mu.T, gamma_diag), x_mu)
            sigma[cluster_index, ...] = sigma_k / normalization_constants[cluster_index]

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
        number_of_observations = X.shape[0]
        number_of_clusters = gamma.shape[1]

        loss_per_observation = np.zeros(number_of_observations)
        for k in range(number_of_clusters):
            loss_per_observation += gamma[:, k] * (np.log(pi[k]) + multivariate_normal.logpdf(X, mean=mu[k, :], cov=sigma[k, ...]))
            loss_per_observation -= gamma[:, k] * np.log(gamma[:, k])

        total_loss = np.sum(loss_per_observation)

        return total_loss

    def train_EM(self, observations, number_of_clusters, rtol=1e-3, max_iter=100, restarts=10):
        '''
        Starts with random initialization *restarts* times
        Runs optimization until saturation with *rtol* reached
        or *max_iter* iterations were made.

        X: (N, d), data points
        C: int, number of clusters
        '''
        number_of_features = observations.shape[1]  # dimension of each object

        best_loss = -1e7
        best_pi = None
        best_mu = None
        best_sigma = None
        best_gamma = None

        for _ in range(restarts):

            pi = 1 / float(number_of_clusters) * np.ones(number_of_clusters)
            mu = np.random.randn(number_of_clusters, number_of_features)
            sigma = np.zeros((number_of_clusters, number_of_features, number_of_features))
            sigma[...] = np.identity(number_of_features)

            gamma = self.E_step(observations, pi, mu, sigma)
            prev_loss = self.compute_vlb(observations, pi, mu, sigma, gamma)

            for _ in tqdm(range(max_iter)):

                gamma = self.E_step(observations, pi, mu, sigma)
                pi, mu, sigma = self.M_step(observations, gamma)
                loss = self.compute_vlb(observations, pi, mu, sigma, gamma)

                if loss / prev_loss < rtol:
                    break

                if loss > best_loss:

                    best_loss = loss
                    best_pi = np.copy(pi)
                    best_mu = np.copy(mu)
                    best_sigma = np.copy(sigma)
                    best_gamma = np.copy(gamma)

                prev_loss = loss

        return best_loss, best_pi, best_mu, best_sigma, best_gamma

