# coding: utf-8
# Script for performing PCA formulated on Grassmann manifold with the sythetic data
#
# Reference: 
# Riemannian Diffusion Adaptation for Distributed Optimization on Manifolds
# Xiuheng Wang, Ricardo Borsoi, Cédric Richard, Ali H. Sayed
#
# 2023/07
# Implemented by
# Xiuheng Wang
# dr.xiuheng.wang@gmail.com

import pymanopt
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

from utils.pca import riemannian_non_cooperative_grassmann, riemannian_diffusion_grassmann,\
										riemannian_centralized_grassmann, grassmann_square_distance
from utils.baselines import DRSGD_pca

# load graph 
A_metropolis = scio.loadmat("data/A_metropolis.mat")["A_metropolis"]
N_vertex = np.shape(A_metropolis)[0]

# experiment setups
N_samples = 1500 # Number of samples for each vertex
dimension = 10 # Dimension of samples
N_components = 5 # Number of components
N_monte_carlo = 10**2 # Number of Monte Carlo experiments

# parameter settings
step_size = 5e-2 # diffusion
step_size_fm = 8e-1 # diffusion
step_size_non_cooperative = step_size # non-cooperative
step_size_centralized = step_size # centralized
beta = 5e-2 # DRSGD
alpha = 8e-1 # DRSGD

# metrics
MSD_cen = np.zeros([N_monte_carlo, N_samples])
MSD_non = np.zeros([N_monte_carlo, N_samples])
MSD_drsgd = np.zeros([N_monte_carlo, N_samples])
MSD_dif = np.zeros([N_monte_carlo, N_samples])
MSD_vertex = np.zeros(N_vertex)

# define manifold
manifold = pymanopt.manifolds.grassmann.Grassmann(dimension, N_components)

# generate random initial points for all agents
initial_points = [manifold.random_point()] * N_vertex

for epoch in range(N_monte_carlo):
	print('Number ' + str(epoch) + " Monte Carlo experiment")

	# generate data following multivariate Gaussian distribution over the graph 
	samples = np.random.multivariate_normal(np.zeros(dimension), np.diag(np.ones(dimension)), size = N_vertex * N_samples)

	# compute the ground truth
	u, sigma, vt = np.linalg.svd(samples, full_matrices=False)
	span_matrix = vt[:N_components, :].T

	# re-generate samples from the ground truth: PCA and SVD are equal with a factor
	sigma = [0.8**i for i in range(dimension)]
	samples = u @ np.diag(sigma) @ vt * np.sqrt(N_vertex * N_samples-1) 

	# shuffle and reshape samples
	np.random.shuffle(samples)
	samples = np.reshape(samples, (N_vertex, N_samples, dimension))

	# Riemannian non-cooperative adaptation
	W_all = riemannian_non_cooperative_grassmann(manifold, A_metropolis, samples, step_size_non_cooperative, initial_points)
	for t in range(N_samples):
		for vertex in range(N_vertex):
			estimated_span_matrix = W_all[t][vertex]
			MSD_vertex[vertex] = grassmann_square_distance(estimated_span_matrix, span_matrix)
		MSD_non[epoch, t] = np.mean(MSD_vertex)

	# DRSGD
	W_all = DRSGD_pca(manifold, A_metropolis, samples, alpha, beta, initial_points)
	for t in range(N_samples):
		for vertex in range(N_vertex):
			estimated_span_matrix = W_all[t][vertex]
			MSD_vertex[vertex] = grassmann_square_distance(estimated_span_matrix, span_matrix)
		MSD_drsgd[epoch, t] = np.mean(MSD_vertex)

	# Riemannian diffusion adaptation
	W_all = riemannian_diffusion_grassmann(manifold, A_metropolis, samples, step_size, step_size_fm, initial_points)
	for t in range(N_samples):
		for vertex in range(N_vertex):
			estimated_span_matrix = W_all[t][vertex]
			MSD_vertex[vertex] = grassmann_square_distance(estimated_span_matrix, span_matrix)
		MSD_dif[epoch, t] = np.mean(MSD_vertex)

	# Riemannian centralized adaptation
	W_all = riemannian_centralized_grassmann(manifold, A_metropolis, samples, step_size_centralized, initial_points)
	for t in range(N_samples):
		estimated_span_matrix = W_all[t]
		MSD_cen[epoch, t] = grassmann_square_distance(estimated_span_matrix, span_matrix)

# mean errros
MSD_non = np.mean(MSD_non, axis=0)
MSD_drsgd = np.mean(MSD_drsgd, axis=0)
MSD_dif = np.mean(MSD_dif, axis=0)
MSD_cen = np.mean(MSD_cen, axis=0)

# draw figures
fig = plt.figure(figsize = (4.8, 3.6), dpi = 150)
plt.plot(10 * np.log10(MSD_non), color="#82B0D2", label='Riemannian non-cooperative')
plt.plot(10 * np.log10(MSD_drsgd), color="#FFBE7A", label='DRSGD')
plt.plot(10 * np.log10(MSD_dif), color="#FA7F6F", label='Riemannian diffusion')
plt.plot(10 * np.log10(MSD_cen), color="#2F7FC1", label='Riemannian centralized')
plt.xlabel("iteration")
plt.ylabel("MSD (dB)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig("figures/MSD_pca.pdf", bbox_inches='tight')
plt.show()

