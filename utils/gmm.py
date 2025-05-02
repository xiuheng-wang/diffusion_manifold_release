import numpy as np
import torch
import pymanopt
from utils.GradientDescent import GradientDescent
from scipy.stats import chi2, ortho_group
from scipy.stats import multivariate_normal
from sklearn.cluster import kmeans_plusplus

def average_log_likelihood(samples, point):
	dimension = np.shape(samples[0])[1]
	samples = np.reshape(samples, (-1, dimension))
	N_samples = np.shape(samples)[0]
	samples = np.concatenate((samples, np.ones((N_samples, 1))), axis=1)
	# estimated solution
	pihat = np.exp(np.concatenate([point[1], [0]], axis=0))
	pihat = pihat / np.sum(pihat)
	N_Gaussian = np.size(pihat)
	muhat = [np.zeros(dimension+1) for _ in range(N_Gaussian)]
	Sigmahat = point[0]
	# # subsample some data from samples: computing the likelihood is expensive
	# subsample_rate = 0.1
	# samples = samples[np.random.choice(N_samples, int(N_samples*subsample_rate), replace=False)]
	# compute average log likelihood
	likelihood = [pihat[ii] * multivariate_normal(muhat[ii], Sigmahat[ii]).pdf(samples) for ii in range(N_Gaussian)]
	log_likelihood = np.log(np.sum(likelihood, axis=0))
	all = np.mean(log_likelihood)
	return all

def initialize_gmm_with_kmeans_plusplus(data, n_components):
	n_samples, n_features = data.shape
	# use kmeans_plusplus to initialize centroids
	centroids, _ = kmeans_plusplus(data, n_clusters=n_components, random_state=1)
	# assign each data point to the nearest centroid
	distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
	labels = np.argmin(distances, axis=1)
	# initialize GMM parameters
	means = centroids
	covariances = []
	for k in range(n_components):
		cluster_data = data[labels == k]
		if len(cluster_data) > 10:
			cov_matrix = np.cov(cluster_data, rowvar=False)
		else:
			cov_matrix = np.eye(n_features)  # use identity if too few points
		covariances.append(cov_matrix)
	covariances = np.array(covariances)
	cluster_sizes = np.bincount(labels, minlength=n_components)
	weights = cluster_sizes / n_samples  # normalize cluster sizes to get probabilities
	return means, covariances, weights

def gmm_parameters(dimension, N_Gaussian):
	nu = 10
	stand_deviation = 3
	max_attempts = 1000
	pi = np.random.gamma(shape=nu, scale=1.0/nu, size=N_Gaussian)
	pi /= np.sum(pi)
	while True:
		Sigma = []
		trace = []
		for _ in range(N_Gaussian):
			eig_vec = ortho_group.rvs(dimension)
			eig_val = chi2.rvs(3.0, size=dimension) # positive
			mtx = eig_vec @ np.diag(eig_val) @ eig_vec.T
			trace.append(np.trace(mtx))
			Sigma.append(mtx)
		mu = [stand_deviation* (np.random.randn(dimension))]
		attempts = 0
		while len(mu) < N_Gaussian:
			vec = stand_deviation * (np.random.randn(dimension))
			if np.all([np.linalg.norm(vec - mu[ii]) >= np.maximum(trace[len(mu)], trace[ii]) for ii in range(len(mu))]):
				mu.append(vec)
			else:
				attempts += 1
				if attempts > max_attempts:  
					# print("Deadlock detected. Restarting...")
					break  
		if len(mu) == N_Gaussian: 
			print("GMM parameters generated!")
			return mu, Sigma, pi

def riemannian_centralized_product(manifold, A, samples, step_size, initial_points):
	num_vertex = np.shape(A)[0]
	# optimizer
	optimizer_sgd = GradientDescent(step_size = step_size, num_iter = 1)
	solution = []
	for t in range(np.shape(samples)[1]):
		# adaptation
		solution_adapt = []
		sample = samples[:, t].squeeze()
		sample = torch.from_numpy(sample)
		@pymanopt.function.pytorch(manifold)
		def cost_adapt(S, v):
			# Unpack parameters 
			nu = torch.cat([v, torch.tensor([0.0])])
			# vals, vecs = torch.linalg.eig(S)
			# vals = torch.where(vals.real>0, vals.real, torch.tensor(1e-6, dtype=torch.float64))
			# S = vecs.real @ torch.diag_embed(vals) @ torch.transpose(vecs.real, -2, -1)
			logdetS = torch.linalg.slogdet(S)[1].unsqueeze(1)
			y = torch.cat([sample.T, torch.ones((1, sample.size()[0]))], dim=0)
			# Calculate log_q
			y = y.unsqueeze(0)
			# 'Probability' of y belonging to each cluster
			log_q = -0.5 * (torch.sum(y * torch.linalg.solve(S, y), dim=1) + logdetS)
			alpha = torch.exp(nu)
			alpha = alpha / alpha.sum() 
			alpha = alpha.unsqueeze(1) 
			loglikvec = torch.logsumexp(torch.log(alpha) + log_q, dim=0)
			return -1.0/num_vertex * torch.sum(loglikvec)
		problem_adapt = pymanopt.Problem(manifold, cost_adapt)
		if t == 0:
			result_adapt = optimizer_sgd.run(problem_adapt, initial_point=initial_points[0])
		else:
			result_adapt = optimizer_sgd.run(problem_adapt, initial_point=solution_combine)
		solution_adapt = result_adapt.point
		solution_combine = solution_adapt
		solution.append(solution_adapt)
	return solution

def riemannian_non_cooperative_product(manifold, A, samples, step_size, initial_points):
	num_vertex = np.shape(A)[0]
	# optimizer
	optimizer_sgd = GradientDescent(step_size = step_size, num_iter = 1)
	solution = []
	for t in range(np.shape(samples)[1]):
		# adaptation
		solution_adapt = []
		for vertex in range(num_vertex):
			sample = samples[vertex, t]
			sample = torch.from_numpy(sample)
			@pymanopt.function.pytorch(manifold)
			def cost_adapt(S, v):
				# Unpack parameters
				nu = torch.cat([v, torch.tensor([0.0])])
				# vals, vecs = torch.linalg.eig(S)
				# vals = torch.where(vals.real>1e-6, vals.real, torch.tensor(1e-6, dtype=torch.float64))
				# S = vecs.real @ torch.diag_embed(vals) @ torch.transpose(vecs.real, -2, -1)
				logdetS = torch.linalg.slogdet(S)[1].unsqueeze(1)
				y = torch.cat([sample.unsqueeze(1), torch.ones((1, 1))], dim=0)
				# Calculate log_q
				y = y.unsqueeze(0)
				# 'Probability' of y belonging to each cluster
				log_q = -0.5 * (torch.sum(y * torch.linalg.solve(S, y), dim=1) + logdetS)
				alpha = torch.exp(nu)
				alpha = alpha / alpha.sum() 
				alpha = alpha.unsqueeze(1) 
				loglikvec = torch.logsumexp(torch.log(alpha) + log_q, dim=0)
				return -loglikvec
			problem_adapt = pymanopt.Problem(manifold, cost_adapt)
			if t == 0:
				result_adapt = optimizer_sgd.run(problem_adapt, initial_point=initial_points[vertex])
			else:
				result_adapt = optimizer_sgd.run(problem_adapt, initial_point=solution_combine[vertex])
			solution_adapt.append(result_adapt.point)
		solution_combine = solution_adapt
		solution.append(solution_adapt)
	return solution
	
def riemannian_diffusion_product(manifold, A, samples, step_size, step_size_fm, initial_points):
	num_vertex = np.shape(A)[0]
	dimension = np.shape(initial_points[0][0])[2]
	# optimizer
	optimizer_sgd = GradientDescent(step_size = step_size, num_iter = 1)
	optimizer_fm = GradientDescent(step_size = step_size_fm, num_iter = 1)
	solution = []
	for t in range(np.shape(samples)[1]):
		# adaptation
		solution_adapt = []
		for vertex in range(num_vertex):
			A_vertex = A[vertex]
			sample = samples[vertex, t]
			sample = torch.from_numpy(sample)
			@pymanopt.function.pytorch(manifold)
			def cost_adapt(S, v):
				# Unpack parameters
				nu = torch.cat([v, torch.tensor([0.0])])
				# vals, vecs = torch.linalg.eig(S)
				# vals = torch.where(vals.real>1e-6, vals.real, torch.tensor(1e-6, dtype=torch.float64))
				# S = vecs.real @ torch.diag_embed(vals) @ torch.transpose(vecs.real, -2, -1)
				logdetS = torch.linalg.slogdet(S)[1].unsqueeze(1)
				y = torch.cat([sample.unsqueeze(1), torch.ones((1, 1))], dim=0)
				# Calculate log_q
				y = y.unsqueeze(0)
				# 'Probability' of y belonging to each cluster
				log_q = -0.5 * (torch.sum(y * torch.linalg.solve(S, y), dim=1) + logdetS)
				alpha = torch.exp(nu)
				alpha = alpha / alpha.sum() 
				alpha = alpha.unsqueeze(1) 
				loglikvec = torch.logsumexp(torch.log(alpha) + log_q, dim=0)
				return -loglikvec
			problem_adapt = pymanopt.Problem(manifold, cost_adapt)
			if t == 0:
				result_adapt = optimizer_sgd.run(problem_adapt, initial_point=initial_points[vertex])
			else:
				result_adapt = optimizer_sgd.run(problem_adapt, initial_point=solution_combine[vertex])
			solution_adapt.append(result_adapt.point)
		# combination
		solution_combine = []
		for vertex in range(num_vertex):
			A_vertex = A[vertex]
			@pymanopt.function.pytorch(manifold)
			def cost_combine(S, v):
				S_est = [solution_adapt[ii][0] for ii in range(num_vertex)]
				temp1 = torch.linalg.eig(torch.DoubleTensor(np.array(S_est)).view(-1, dimension, dimension))
				# vals = torch.where(temp1[0].real > 1e-6, temp1[0].real, torch.tensor(1e-6, dtype=torch.float64))
				vals = temp1[0].real + 1e-6
				c = torch.bmm(torch.bmm(temp1[1].real, \
							torch.diag_embed(torch.sqrt(1/vals), offset=0, dim1=-2, dim2=-1)),\
							torch.transpose(temp1[1].real, 1, 2))
				temp2 = torch.bmm(torch.bmm(c, S.unsqueeze(0).repeat(num_vertex, 1, 1, 1).view(-1, dimension, dimension)), c)
				temp3 = torch.log(torch.linalg.eig(temp2)[0])
				temp4 = temp3.unsqueeze(0).view(num_vertex, -1)
				v_est = [solution_adapt[ii][1] for ii in range(num_vertex)]
				cost = torch.norm(temp4, dim=1)**2 + torch.norm(torch.DoubleTensor(np.array(v_est))-v.unsqueeze(0).repeat(num_vertex, 1), dim=1)**2
				return torch.sum(torch.from_numpy(A_vertex) * cost)
			problem_combine = pymanopt.Problem(manifold, cost_combine)
			result_combine = optimizer_fm.run(problem_combine, initial_point=solution_adapt[vertex])
			solution_combine.append(result_combine.point)
		solution.append(solution_combine)
	return solution

