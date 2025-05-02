# coding: utf-8
# Script for performing PCA formulated on Grassmann manifold with the mnist dataset
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
import time

from utils.pca import riemannian_non_cooperative_grassmann, riemannian_diffusion_grassmann,\
										riemannian_centralized_grassmann, grassmann_square_distance
from utils.baselines import DRSGD_pca

# load graph 
A_metropolis = scio.loadmat("data/A_metropolis.mat")["A_metropolis"]
N_vertex = np.shape(A_metropolis)[0]

# load mnist dataset
mnist = scio.loadmat("data/mnist-original.mat")
samples = mnist["data"].T / 255.0 # normalized
samples -= samples.mean(axis=0) # centralized
np.random.seed(1)
np.random.shuffle(samples)
N_samples = int(np.shape(samples)[0] / N_vertex) # Number of samples for each vertex
dimension = np.shape(samples)[1] # Dimension of samples
N_components = 5 # Number of components

# parameter settings
step_size = 2e-3 # diffusion
step_size_fm = 5e-3 # diffusion
step_size_non_cooperative = 2.5e-3 # non-cooperative
step_size_centralized = step_size # centralized
beta = 2e-3 # DRSGD
alpha = 1e-3 # DRSGD

# metrics
MSD = np.zeros(N_samples)
MSD_vertex = np.zeros(N_vertex)

# compute the ground truth: PCA and SVD are equal with a factor
u, sigma, vt = np.linalg.svd(samples / np.sqrt(N_vertex * N_samples-1), full_matrices=False)
span_matrix = vt[:N_components, :].T

# partitioned in to subsets
samples = np.reshape(samples, (N_vertex, N_samples, dimension)) 

# define manifold
manifold = pymanopt.manifolds.grassmann.Grassmann(dimension, N_components)

# generate random initial points for all agents
initial_points = [manifold.random_point() for _ in range(N_vertex)]

fig = plt.figure(figsize = (4.8, 3.6), dpi = 150)

# Riemannian non-cooperative adaptation
start = time.time()
W_all = riemannian_non_cooperative_grassmann(manifold, A_metropolis, samples, step_size_non_cooperative, initial_points)
end = time.time()
print('Running time of Riemannian non-cooperative adaptation: ', end - start)
for t in range(N_samples):
	for vertex in range(N_vertex):
		estimated_span_matrix = W_all[t][vertex]
		MSD_vertex[vertex] = grassmann_square_distance(estimated_span_matrix, span_matrix)
	MSD[t] = np.mean(MSD_vertex)
plt.plot(10 * np.log10(MSD), color="#82B0D2", label='Riemannian non-cooperative')

# DRSGD
start = time.time()
W_all = DRSGD_pca(manifold, A_metropolis, samples, alpha, beta, initial_points)
end = time.time()
print('Running time of DRSGD: ', end - start)
for t in range(N_samples):
	for vertex in range(N_vertex):
		estimated_span_matrix = W_all[t][vertex]
		MSD_vertex[vertex] = grassmann_square_distance(estimated_span_matrix, span_matrix)
	MSD[t] = np.mean(MSD_vertex)
plt.plot(10 * np.log10(MSD), color="#FFBE7A", label='DRSGD')

# Riemannian diffusion adaptation
start = time.time()
W_all = riemannian_diffusion_grassmann(manifold, A_metropolis, samples, step_size, step_size_fm, initial_points)
end = time.time()
print('Running time of Riemannian diffusion adaptation: ', end - start)
for t in range(N_samples):
	for vertex in range(N_vertex):
		estimated_span_matrix = W_all[t][vertex]
		MSD_vertex[vertex] = grassmann_square_distance(estimated_span_matrix, span_matrix)
	MSD[t] = np.mean(MSD_vertex)
plt.plot(10 * np.log10(MSD), color="#FA7F6F", label='Riemannian diffusion')

# Riemannian centralized adaptation
start = time.time()
W_all = riemannian_centralized_grassmann(manifold, A_metropolis, samples, step_size_centralized, initial_points)
end = time.time()
print('Running time of Riemannian centralized adaptation: ', end - start)
for t in range(N_samples):
	estimated_span_matrix = W_all[t]
	MSD[t] = grassmann_square_distance(estimated_span_matrix, span_matrix)
plt.plot(10 * np.log10(MSD), color="#2F7FC1", label='Riemannian centralized')

# draw figures
plt.xlabel("iteration")
plt.ylabel("MSD (dB)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig("figures/MSD_pca_mnist.pdf", bbox_inches='tight')
plt.show()

