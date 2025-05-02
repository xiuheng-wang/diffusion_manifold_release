import numpy as np
import torch
import pymanopt
from utils.GradientDescent import GradientDescent

def grassmann_square_distance(matrix, point):
	temp1 = torch.from_numpy(matrix.transpose()) @ point
	temp2 = torch.linalg.svd(temp1)[1]
	temp3 = torch.acos(torch.clamp(temp2, -1+1e-8, 1-1e-8))
	return torch.norm(temp3)**2

def riemannian_centralized_grassmann(manifold, A, samples, step_size, initial_points):
	num_vertex = np.shape(A)[0]
	# optimizer
	optimizer_sgd = GradientDescent(step_size = step_size, num_iter = 1)
	solution = []
	for t in range(np.shape(samples)[1]):
		# adaptation
		solution_adapt = []
		sample = samples[:, t].squeeze()
		matrix = sample.T @ sample / num_vertex
		matrix_ = torch.from_numpy(matrix)
		@pymanopt.function.pytorch(manifold)
		def cost_adapt(w):
			return - torch.trace(torch.transpose(w, 1, 0) @ matrix_ @ w)
		problem_adapt = pymanopt.Problem(manifold, cost_adapt)
		if t == 0:
			result_adapt = optimizer_sgd.run(problem_adapt, initial_point=initial_points[0])
		else:
			result_adapt = optimizer_sgd.run(problem_adapt, initial_point=solution_combine)
		solution_adapt = result_adapt.point
		solution_combine = solution_adapt
		solution.append(solution_adapt)
	return solution

def riemannian_non_cooperative_grassmann(manifold, A, samples, step_size, initial_points):
	num_vertex = np.shape(A)[0]
	# optimizer
	optimizer_sgd = GradientDescent(step_size = step_size, num_iter = 1)
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
		solution_combine = solution_adapt
		solution.append(solution_adapt)
	return solution

def riemannian_diffusion_grassmann(manifold, A, samples, step_size, step_size_fm, initial_points):
	num_vertex = np.shape(A)[0]
	# optimizer
	optimizer_sgd = GradientDescent(step_size = step_size, num_iter = 1)
	optimizer_fm = GradientDescent(step_size = step_size_fm, num_iter = 1) # compute weighted Frechet mean
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
			result_combine = optimizer_fm.run(problem_combine, initial_point=solution_adapt[vertex])
			solution_combine.append(result_combine.point)
		solution.append(solution_combine)
	return solution

