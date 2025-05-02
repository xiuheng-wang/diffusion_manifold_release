# coding: utf-8
# Script for performing GMM inference formulated on the product manifold with the sythetic data
#
# Reference: 
# Riemannian Diffusion Adaptation for Distributed Optimization on Manifolds
# Xiuheng Wang, Ricardo Borsoi, Cédric Richard, Ali H. Sayed
#
# 2024/07
# Implemented by
# Xiuheng Wang
# dr.xiuheng.wang@gmail.com

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

from pymanopt.manifolds import SymmetricPositiveDefinite, Euclidean, Product
from utils.gmm import riemannian_non_cooperative_product, riemannian_diffusion_product, riemannian_centralized_product,\
							gmm_parameters, average_log_likelihood, initialize_gmm_with_kmeans_plusplus
from utils.baselines import DRSGD_gmm

# load graph 
A_metropolis = scio.loadmat("data/A_metropolis.mat")["A_metropolis"]
N_vertex = np.shape(A_metropolis)[0]

# experiment setups
N_samples = 1500 # Number of samples for each vertex
dimension = 8 # Dimension of samples
N_Gaussian = 3 # Number of Gaussian
N_monte_carlo = 10**1 # Number of Monte Carlo experiments

# parameter settings
step_size = 4e-2 # diffusion
step_size_fm = 5e-2 # Frechet mean
step_size_non_cooperative = step_size # non-cooperative
step_size_centralized = step_size # centralized
beta = 4e-2 # DRSGD
alpha = 5e-2 # DRSGD

# metrics
ALL_cen = np.zeros([N_monte_carlo, N_samples])
ALL_non = np.zeros([N_monte_carlo, N_samples])
ALL_drsgd = np.zeros([N_monte_carlo, N_samples])
ALL_dif = np.zeros([N_monte_carlo, N_samples])
ALL_vertex = np.zeros(N_vertex)

# define manifold
manifold_spd = SymmetricPositiveDefinite(dimension, k=N_Gaussian)
manifold_euclidean = Euclidean(N_Gaussian - 1)
manifold = Product([SymmetricPositiveDefinite(dimension + 1, k=N_Gaussian), Euclidean(N_Gaussian - 1)])

for epoch in range(N_monte_carlo):
    print('Number ' + str(epoch) + " Monte Carlo experiment")

    # generate the parameters of Gaussian mixture model
    mu, Sigma, pi = gmm_parameters(dimension, N_Gaussian)
    S = [np.block([[Sigma[ii] + np.outer(mu[ii], mu[ii]), mu[ii].reshape(-1, 1)], [mu[ii].reshape(1, -1), np.array([[1]])]]) for ii in range(N_Gaussian)]
    v = [np.log(pi[ii]/pi[-1]) for ii in range(N_Gaussian-1)]
    parameter_matrix = [S, v]

    # generate data following Gaussian mixture model over the graph
    components = np.random.choice(N_Gaussian, size=N_samples*N_vertex, p=pi)
    samples = np.zeros((N_samples*N_vertex, dimension))
    for k in range(N_Gaussian):
        # indices of current component in X
        indices = k == components
        # number of those occurrences
        n_k = indices.sum()
        if n_k > 0:
            samples[indices, :] = np.random.multivariate_normal(mu[k], Sigma[k], n_k)

    # generate random initial points for all agents: mean
    mu, Sigma, pi = initialize_gmm_with_kmeans_plusplus(samples, N_Gaussian)
    S = [np.block([[Sigma[ii] + np.outer(mu[ii], mu[ii]), mu[ii].reshape(-1, 1)], [mu[ii].reshape(1, -1), np.array([[1]])]]) for ii in range(N_Gaussian)]
    v = [np.log(pi[ii]/pi[-1]) for ii in range(N_Gaussian-1)]
    initial_points = [[np.array(S), np.array(v)]] * N_vertex

    # shuffle and reshape samples
    np.random.shuffle(samples)
    samples = np.reshape(samples, (N_vertex, N_samples, dimension))

    # compute the optimal ALL
    ALL_opt = average_log_likelihood(samples, parameter_matrix)

    # Riemannian non-cooperative adaptation
    W_all = riemannian_non_cooperative_product(manifold, A_metropolis, samples, step_size_non_cooperative, initial_points)
    for t in range(N_samples):
        for vertex in range(N_vertex):
            estimated_parameter_matrix = W_all[t][vertex]
            ALL_vertex[vertex] = average_log_likelihood(samples, estimated_parameter_matrix)
        ALL_non[epoch, t] = ALL_opt - np.mean(ALL_vertex)

    # DRSGD
    W_all = DRSGD_gmm(manifold, A_metropolis, samples, alpha, beta, initial_points)
    for t in range(N_samples):
        for vertex in range(N_vertex):
            estimated_parameter_matrix = W_all[t][vertex]
            ALL_vertex[vertex] = average_log_likelihood(samples, estimated_parameter_matrix)
        ALL_drsgd[epoch, t] = ALL_opt - np.mean(ALL_vertex)

    # Riemannian diffusion adaptation
    W_all = riemannian_diffusion_product(manifold, A_metropolis, samples, step_size, step_size_fm, initial_points)
    for t in range(N_samples):
        for vertex in range(N_vertex):
            estimated_parameter_matrix = W_all[t][vertex]
            ALL_vertex[vertex] = average_log_likelihood(samples, estimated_parameter_matrix)
        ALL_dif[epoch, t] = ALL_opt - np.mean(ALL_vertex)

    # Riemannian centralized adaptation
    W_all = riemannian_centralized_product(manifold, A_metropolis, samples, step_size_centralized, initial_points)
    for t in range(N_samples):
        estimated_parameter_matrix = W_all[t]
        ALL_cen[epoch, t] = ALL_opt - average_log_likelihood(samples, estimated_parameter_matrix)

# mean errros
ALL_non = np.mean(ALL_non, axis=0)
ALL_drsgd = np.mean(ALL_drsgd, axis=0)
ALL_dif = np.mean(ALL_dif, axis=0)
ALL_cen = np.mean(ALL_cen, axis=0)

# draw figures
fig = plt.figure(figsize = (4.8, 3.6), dpi = 150)
plt.plot(10 * np.log10(ALL_non), color="#82B0D2", label='Riemannian non-cooperative')
plt.plot(10 * np.log10(ALL_drsgd), color="#FFBE7A", label='ECGMM')
plt.plot(10 * np.log10(ALL_dif), color="#FA7F6F", label='Riemannian diffusion')
plt.plot(10 * np.log10(ALL_cen), color="#2F7FC1", label='Riemannian centralized')
plt.xlabel("iteration")
plt.ylabel(r"$ALL^* - $" + "ALL (dB)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig("figures/ALL_gmm.pdf", bbox_inches='tight')
plt.show()
