import torch
def rejection_sampling(points, density, target_number_samples):
	uniforms = torch.rand(points.shape[0])
	keeping_probability = target_number_samples * torch.tensor(density) / density.sum()

	keep = uniforms < keeping_probability
	points = points[keep]
	density = density[keep]
	
	return points, density, keep