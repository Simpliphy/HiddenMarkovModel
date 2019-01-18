import numpy as np
import matplotlib.pyplot as plt

from hmm.GaussianSoftClustering import GaussianSoftClustering

if __name__=="__main__":

    samples = np.load('./samples.npz')

    X = samples['data']
    pi0 = samples['pi0']
    mu0 = samples['mu0']
    sigma0 = samples['sigma0']
    plt.scatter(X[:, 0], X[:, 1], c='grey', s=30)
    plt.axis('equal')
    plt.show()

    gaussian_clustering_model = GaussianSoftClustering()

    pi, mu, sigma = pi0, mu0, sigma0
    gamma = gaussian_clustering_model.E_step(X, pi, mu, sigma)
    pi, mu, sigma = gaussian_clustering_model.M_step(X, gamma)
    loss = gaussian_clustering_model.compute_vlb(X, pi, mu, sigma, gamma)

    best_loss, best_pi, best_mu, best_sigma, best_gamma = gaussian_clustering_model.train_EM(X, 3, restarts=3)

    gamma = gaussian_clustering_model.E_step(X, best_pi, best_mu, best_sigma)

    labels = gamma.argmax(1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30)
    plt.axis('equal')
    plt.show()
