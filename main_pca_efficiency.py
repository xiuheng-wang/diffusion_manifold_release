# coding: utf-8
# Script for comparing the performance and running time of (inefficient) Riemannian diffusion algorithms
#
# Reference: 
# Riemannian Diffusion Adaptation for Distributed Optimization on Manifolds
# Xiuheng Wang, Ricardo Borsoi, Cédric Richard, Ali H. Sayed
#
# 2025/03
# Implemented by
# Xiuheng Wang
# dr.xiuheng.wang@gmail.com

import pymanopt
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

from utils.pca import riemannian_diffusion_grassmann, grassmann_square_distance
from utils.baselines import inefficient_riemannian_diffusion_grassmann
import time

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

# metrics
MSD_idif = np.zeros([N_monte_carlo, N_samples])
MSD_dif = np.zeros([N_monte_carlo, N_samples])
MSD_vertex = np.zeros(N_vertex)

# time
time_idif = np.zeros([N_monte_carlo])
time_dif = np.zeros([N_monte_carlo])

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

	# Inefficient Riemannian diffusion adaptation
	start_time = time.time()
	W_all = inefficient_riemannian_diffusion_grassmann(manifold, A_metropolis, samples, step_size, initial_points)
	for t in range(N_samples):
		for vertex in range(N_vertex):
			estimated_span_matrix = W_all[t][vertex]
			MSD_vertex[vertex] = grassmann_square_distance(estimated_span_matrix, span_matrix)
		MSD_idif[epoch, t] = np.mean(MSD_vertex)
	time_idif[epoch] = time.time() - start_time

	# Riemannian diffusion adaptation
	start_time = time.time()
	W_all = riemannian_diffusion_grassmann(manifold, A_metropolis, samples, step_size, step_size_fm, initial_points)
	for t in range(N_samples):
		for vertex in range(N_vertex):
			estimated_span_matrix = W_all[t][vertex]
			MSD_vertex[vertex] = grassmann_square_distance(estimated_span_matrix, span_matrix)
		MSD_dif[epoch, t] = np.mean(MSD_vertex)
	time_dif[epoch] = time.time() - start_time

# mean errros
MSD_idif = np.mean(MSD_idif, axis=0)
MSD_dif = np.mean(MSD_dif, axis=0)

# draw figures
fig = plt.figure(figsize = (4.8, 3.6), dpi = 150)
plt.plot(10 * np.log10(MSD_idif), color="#FFBE7A", label='Inefficient Riemannian diffusion ' + '(' + f'{np.mean(time_idif) / N_vertex:.2f}' + 's)')
plt.plot(10 * np.log10(MSD_dif), color="#FA7F6F", label='Riemannian diffusion ' + '(' + f'{np.mean(time_dif) / N_vertex:.2f}' + 's)')
plt.xlabel("iteration")
plt.ylabel("MSD (dB)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig("figures/MSD_pca_efficiency.pdf", bbox_inches='tight')
plt.show()

