# coding: utf-8
# Script for performing GMM inference formulated on the product manifold with the mnist dataset
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
import time

from pymanopt.manifolds import SymmetricPositiveDefinite, Euclidean, Product
from utils.gmm import riemannian_non_cooperative_product, riemannian_diffusion_product, riemannian_centralized_product,\
							average_log_likelihood, initialize_gmm_with_kmeans_plusplus
from utils.baselines import DRSGD_gmm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# load graph 
A_metropolis = scio.loadmat("data/A_metropolis.mat")["A_metropolis"]
N_vertex = np.shape(A_metropolis)[0]

# load mnist dataset
mnist = scio.loadmat("data/mnist-original.mat")
samples = mnist["data"].T / 225.0 # normalized
np.random.seed(1)
np.random.shuffle(samples)
samples = PCA(n_components=20).fit_transform(samples) # 26 dimensions

N_samples = int(np.shape(samples)[0] / N_vertex) # Number of samples for each vertex
dimension = np.shape(samples)[1] # Dimension of samples
N_Gaussian = 7 # Number of Gaussian

# parameter settings
step_size = 8e-2 # diffusion
step_size_fm = 8e-2 # Frechet mean
step_size_non_cooperative = step_size # non-cooperative
step_size_centralized = step_size # centralized
beta = 8e-2 # DRSGD
alpha = 8e-2 # DRSGD

# metrics
ALL = np.zeros(N_samples)
ALL_vertex = np.zeros(N_vertex)

# define manifold
manifold = Product([SymmetricPositiveDefinite(dimension + 1, k=N_Gaussian), Euclidean(N_Gaussian - 1)])

# generate initial points for all agents
mu, Sigma, pi = initialize_gmm_with_kmeans_plusplus(samples, N_Gaussian)
v = [np.log(pi[ii]/pi[-1]) for ii in range(N_Gaussian-1)]
S = [np.block([[Sigma[ii] + np.outer(mu[ii], mu[ii]), mu[ii].reshape(-1, 1)], [mu[ii].reshape(1, -1), np.array([[1]])]]) for ii in range(N_Gaussian)]
initial_points = [[np.array(S), np.array(v)]] * N_vertex

# compute the ground truth by EM algorithm
GMM = GaussianMixture(n_components=N_Gaussian, max_iter=1000, init_params='k-means++').fit(samples)
mu, Sigma, pi = GMM.means_, GMM.covariances_, GMM.weights_
S = [np.block([[Sigma[ii] + np.outer(mu[ii], mu[ii]), mu[ii].reshape(-1, 1)], [mu[ii].reshape(1, -1), np.array([[1]])]]) for ii in range(N_Gaussian)]
v = [np.log(pi[ii]/pi[-1]) for ii in range(N_Gaussian-1)]
parameter_matrix = [S, v]

# partitioned in to subsets
samples = np.reshape(samples, (N_vertex, N_samples, dimension))

# compute the optimal ALL
ALL_opt = average_log_likelihood(samples, parameter_matrix)

fig = plt.figure(figsize = (4.8, 3.6), dpi = 150)

# Riemannian non-cooperative adaptation
start = time.time()
W_all = riemannian_non_cooperative_product(manifold, A_metropolis, samples, step_size_non_cooperative, initial_points)
end = time.time()
print('Running time of Riemannian non-cooperative adaptation: ', end - start)
for t in range(N_samples):
	for vertex in range(N_vertex):
		estimated_parameter_matrix = W_all[t][vertex]
		ALL_vertex[vertex] = average_log_likelihood(samples, estimated_parameter_matrix)
	ALL[t] = np.mean(ALL_vertex)
plt.plot(10 * np.log10(ALL_opt - ALL), color="#82B0D2", label='Riemannian non-cooperative')

# DRSGD
start = time.time()
W_all = DRSGD_gmm(manifold, A_metropolis, samples, alpha, beta, initial_points)
end = time.time()
print('Running time of ECGMM: ', end - start)
for t in range(N_samples):
	for vertex in range(N_vertex):
		estimated_parameter_matrix = W_all[t][vertex]
		ALL_vertex[vertex] = average_log_likelihood(samples, estimated_parameter_matrix)
	ALL[t] = np.mean(ALL_vertex)
plt.plot(10 * np.log10(ALL_opt - ALL), color="#FFBE7A", label='ECGMM')

# Riemannian diffusion adaptation
start = time.time()
W_all = riemannian_diffusion_product(manifold, A_metropolis, samples, step_size, step_size_fm, initial_points)
end = time.time()
print('Running time of Riemannian diffusion adaptation: ', end - start)
for t in range(N_samples):
	for vertex in range(N_vertex):
		estimated_parameter_matrix = W_all[t][vertex]
		ALL_vertex[vertex] = average_log_likelihood(samples, estimated_parameter_matrix)
	ALL[t] = np.mean(ALL_vertex)
plt.plot(10 * np.log10(ALL_opt - ALL), color="#FA7F6F", label='Riemannian diffusion')

# Riemannian centralized adaptation
start = time.time()
W_all = riemannian_centralized_product(manifold, A_metropolis, samples, step_size_centralized, initial_points)
end = time.time()
print('Running time of Riemannian centralized adaptation: ', end - start)
for t in range(N_samples):
	estimated_parameter_matrix = W_all[t]
	ALL[t] = average_log_likelihood(samples, estimated_parameter_matrix)
plt.plot(10 * np.log10(ALL_opt - ALL), color="#2F7FC1", label='Riemannian centralized')

# draw figures
plt.xlabel("iteration")
plt.ylabel(r"$ALL^* - $" + "ALL (dB)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig("figures/ALL_gmm_mnist.pdf", bbox_inches='tight')
plt.show()
