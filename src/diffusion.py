# contains functions for diffusion processes

from tqdm import tqdm
import torch
import math

# FORWARD DIFFUSION RELATED

def get_linear_schedule_forward(beta_1, beta_T, num_steps):
	"""
	return linear schedule for forward diffusion (beta)
	"""

	step = (beta_T - beta_1) / num_steps

	beta_schedule = [beta_1 + (idx * step) for idx in range(1, num_steps+1)]

	return beta_schedule

def forward_diffusion_single_step(x_t_minus_1, beta_t, device, gaussian_noise=None):
	"""
	perform a single diffusion step and return the resultant image // no clipping
	"""

	if gaussian_noise is None:
		gaussian_noise = torch.normal(mean=torch.zeros(x_t_minus_1.shape)).to(device)

	x_t = (torch.sqrt(torch.tensor(1 - beta_t)) * x_t_minus_1) + (torch.sqrt(torch.tensor(beta_t)) * gaussian_noise)

	return x_t, gaussian_noise

def forward_diffusion_linear_schedule(x_0, beta_1, beta_T, num_steps, device):
	"""
	apply forward diffusion step num_step times with a linear schedule
	"""

	beta_schedule = get_linear_schedule_forward(beta_1, beta_T, num_steps)

	diffusion_path = [x_0]

	for idx in tqdm(range(1, num_steps+1), desc='Forward Diffusion'):

		diffusion_path.append(forward_diffusion_single_step(diffusion_path[-1], beta_schedule[idx-1], device)[0])

	return diffusion_path

def forward_diffusion_step_t(x_0, beta_1, beta_T, num_steps, time_step, device, gaussian_noise=None):
	"""
	return the noisy image at step t using direct formulation
	instead of applying the gaussian noise t times
	https://huggingface.co/blog/annotated-diffusion
	"""

	_, alpha_bar_schedule = get_linear_schedule_reverse(beta_1, beta_T, num_steps)

	return forward_diffusion_single_step(x_0, 1-alpha_bar_schedule[time_step-1], device, gaussian_noise)[0]


# REVERSE DIFFUSION RELATED

def get_linear_schedule_reverse(beta_1, beta_T, num_steps):
	"""
	return linear schedule for backward diffusion (alpha and alpha_bar)
	"""

	beta_schedule = get_linear_schedule_forward(beta_1, beta_T, num_steps)

	alpha_schedule = [1-b for b in beta_schedule]

	alpha_bar_schedule = [a for a in alpha_schedule]

	for i in range(1, len(alpha_bar_schedule)):
		alpha_bar_schedule[i] *= alpha_bar_schedule[i-1]

	return alpha_schedule, alpha_bar_schedule

def reverse_diffusion_single_step(denoise_model, x_t, alpha_t, alpha_bar_t, t, device):
	"""
	apply the denoising model for a single step and return the resulting image
	"""

	with torch.no_grad():

		predicted_noise = denoise_model(x_t, torch.tensor([t]).to(device))

		scaled_noise = ((1 - alpha_t) / (math.sqrt(1 - alpha_bar_t))) * predicted_noise

		x_t_minus_1 = (x_t - scaled_noise) / math.sqrt(alpha_t)

		# calculation
		# sigma_t = sqrt(beta_t)
		# sigma_t = sqrt(1 - alpha_t)

		if t > 1:

			gaussian_noise = torch.normal(mean=torch.zeros(x_t_minus_1.shape)).to(device)
			x_t_minus_1 += gaussian_noise * math.sqrt(1 - alpha_t)

	return x_t_minus_1, predicted_noise

def reverse_diffusion_linear_schedule(denoise_model, x_T, beta_1, beta_T, num_steps, device):
	"""
	apply reverse diffusion step num_step times with a linear schedule
	"""

	alpha_schedule, alpha_bar_schedule = get_linear_schedule_reverse(beta_1, beta_T, num_steps)

	diffusion_path = [x_T]

	for idx in tqdm(range(num_steps, 0, -1), desc='Reverse Diffusion'):

		diffusion_path.append(reverse_diffusion_single_step(denoise_model, diffusion_path[-1], 
								alpha_schedule[idx-1], alpha_bar_schedule[idx-1], idx, device)[0])

	return diffusion_path
