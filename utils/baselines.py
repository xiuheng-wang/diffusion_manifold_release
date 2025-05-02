import numpy as np
import torch
import pymanopt
from utils.GradientDescent import GradientDescent
from utils.SteepestDescent import SteepestDescent

def proj_tangent(x, d):
		xd = np.matmul(x.T,  d)
		pd = d - 0.5 * np.matmul(x, xd + xd.T)
		return pd

def consensus_pca(a, x):
	return sum(weight * matrix for weight, matrix in zip(a, x))

def DRSGD_pca(manifold, A, samples, alpha, beta, initial_points):
	num_vertex = np.shape(A)[0]
	solution = []
	solution_old = initial_points
	for t in range(np.shape(samples)[1]):
		# adaptation
		solution_new = []
		for vertex in range(num_vertex):
			A_vertex = A[vertex]
			sample = samples[vertex, t][np.newaxis, :]
			matrix = sample.T @ sample
			matrix_ = torch.from_numpy(matrix)
			@pymanopt.function.pytorch(manifold)
			def cost(w):
				return - torch.trace(torch.transpose(w, 1, 0) @ matrix_ @ w)
			problem = pymanopt.Problem(manifold, cost)
			gradient = problem.riemannian_gradient
			x = solution_old[vertex]
			grad = gradient(x)
			x = manifold.retraction(x, alpha*proj_tangent(x, consensus_pca(A_vertex, solution_old)-x) - beta*grad)
			solution_new.append(x)
		solution_old = solution_new
		solution.append(solution_new)
	return solution

def consensus_gmm(a, x):
	S = [x[ii][0] for ii in range(np.size(a))]
	v = [x[ii][1] for ii in range(np.size(a))]
	return [sum(weight * matrix for weight, matrix in zip(a, S)), sum(weight * matrix for weight, matrix in zip(a, v))]

def DRSGD_gmm(manifold, A, samples, alpha, beta, initial_points):
	num_vertex = np.shape(A)[0]
	solution = []
	solution_old = initial_points
	for t in range(np.shape(samples)[1]):
		# adaptation
		solution_new = []
		for vertex in range(num_vertex):
			A_vertex = A[vertex]
			sample = samples[vertex, t]
			sample = torch.from_numpy(sample)
			@pymanopt.function.pytorch(manifold)
			def cost(S, v):
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
			problem = pymanopt.Problem(manifold, cost)
			gradient = problem.riemannian_gradient
			x = solution_old[vertex]
			grad = gradient(x)
			x = manifold.retraction(x, alpha*manifold.projection(x, [a - b for a, b in zip(consensus_gmm(A_vertex, solution_old), x
			)]) - beta*grad)
			solution_new.append(x)
		solution_old = solution_new
		solution.append(solution_new)
	return solution

def inefficient_riemannian_diffusion_grassmann(manifold, A, samples, step_size, initial_points):
	num_vertex = np.shape(A)[0]
	# optimizer
	optimizer_sgd = GradientDescent(step_size = step_size, num_iter = 1)
	optimizer_wfm = SteepestDescent(num_iter = 10) # compute weighted Frechet mean
	solution = []
	for t in range(np.shape(samples)[1]):
		# adaptation
		solution_adapt = []
		for vertex in range(num_vertex):
			sample = samples[vertex, t][np.newaxis, :]
			matrix = sample.T @ sample
			matrix_ = torch.from_numpy(matrix)
			@pymanopt.function.pytorch(manifold)
			def cost_adapt(w):
				return - torch.trace(torch.transpose(w, 1, 0) @ matrix_ @ w)
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
			def cost_combine(point):
				temp1 = torch.bmm(torch.transpose(torch.DoubleTensor(np.array(solution_adapt)), 1, 2),\
									point.unsqueeze(0).repeat(num_vertex, 1, 1))
				temp2 = torch.linalg.svd(temp1)[1]
				temp3 = torch.acos(torch.clamp(temp2, -1+1e-8, 1-1e-8))
				cost = torch.norm(temp3, dim=1)**2
				return torch.sum(torch.from_numpy(A_vertex) * cost)
			problem_combine = pymanopt.Problem(manifold, cost_combine)
			result_combine = optimizer_wfm.run(problem_combine, initial_point=solution_adapt[vertex])
			solution_combine.append(result_combine.point)
		solution.append(solution_combine)
	return solution

