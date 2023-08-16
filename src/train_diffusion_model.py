# this file contains the training routine

from argparse import ArgumentParser
import json5
import os
from pathlib import Path
import shutil

import torch
from denoising_diffusion_pytorch import Unet
import math
import random
from torch.optim import Adam

from utils import *
from diffusion import *

def train(input_image_dir, output_dir, config):
	"""
	training routine
	"""

	# get the device to be used
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print(device)

	# read the input images as tensors
	all_images = [os.path.join(input_image_dir, img) for img in os.listdir(input_image_dir)]
	img_tensor = [read_image_tensor(img, image_size=config["image_size"]).to(device) for img in all_images]

	# save the image just to check
	save_image_tensor(img_tensor[0], os.path.join(output_dir, "img/input_image_tensor.jpg"))

	# enter training hyper parameters here
	# get the training hyper parameters from config

	beta_1 = config["beta_1"] # this is actually beta zero
	beta_T = config["beta_T"]
	num_diffusion_steps = config["num_diffusion_steps"]
	num_epochs = config["num_epochs"]
	learning_rate = config["learning_rate"]
	batch_size = config["batch_size"]

	# declare the model here
	denoise_model = Unet(dim=config["model_dim"], dim_mults = config["model_dim_mults"]).to(device).train()

	# declare the optimizer here
	optimizer = Adam(denoise_model.parameters(), lr=learning_rate)

	# start the training process
	# initialize timestep sampling frequencies
	timestep_frequency = {idx: 0 for idx in range(1, num_diffusion_steps+1)}

	# declare the loss function
	loss_fxn = torch.nn.MSELoss()

	# unsqueenze the image as a batch
	img_tensor = torch.stack(img_tensor * (batch_size // len(img_tensor)))
	print(img_tensor.shape)

	# iterate for num_epochs

	for epoch in tqdm(range(1, num_epochs+1), desc="Training"):

		# sample a uniformly random time step t
		t = random.randint(1, num_diffusion_steps)
		timestep_frequency[t] += 1

		# sample random gaussion noise (target)
		target_noise = torch.normal(mean=torch.zeros(img_tensor.shape)).to(device)

		# get the noisy image at timestep t
		x_t = forward_diffusion_step_t(img_tensor, beta_1, beta_T, num_diffusion_steps, t, device, target_noise)

		# get the model output
		model_noise = denoise_model(x_t, torch.tensor([t]).to(device))

		# calculate the loss
		loss = loss_fxn(model_noise, target_noise)

		# compute the gradients, update the weights, flush the gradients
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		if epoch % config["report_every_steps"] == 0:
			print(epoch, loss)

			if epoch % config["save_every_steps"] == 0:
				
				# save the model checkpount
				torch.save(denoise_model, os.path.join(output_dir, f"ckpt/step_latest.pt"))

				# generate image path for this timestep
				x_T = torch.normal(mean=torch.zeros(img_tensor[0].squeeze().shape)).to(device).unsqueeze(0)

				# denoise_model.val()
				
				denoised_path = reverse_diffusion_linear_schedule(denoise_model, x_T, beta_1, beta_T, num_diffusion_steps, device)
				save_image_tensor(denoised_path[num_diffusion_steps][0], os.path.join(output_dir, f"img/sampled_image_step_{epoch}.jpg"))
				
				# denoise_model.train()
				del denoised_path

	print(timestep_frequency)

	torch.save(denoise_model, os.path.join(output_dir, f"ckpt/step_latest.pt"))

	# generate images for this timestep
	x_T = torch.normal(mean=torch.zeros(img_tensor[0].squeeze().shape)).to(device).unsqueeze(0)
	# denoise_model.val()
	denoised_path = reverse_diffusion_linear_schedule(denoise_model, x_T, beta_1, beta_T, num_diffusion_steps, device)
	save_image_tensor(denoised_path[num_diffusion_steps][0], os.path.join(output_dir, f"img/sampled_image_step_final.jpg"))
	# denoise_model.train()
	del denoised_path

	return


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('--input_image_dir', type=str, help='path to input images dir to be used for training')
	parser.add_argument('--output_dir', type=str, help='path to output directory')
	parser.add_argument('--config_file', type=str, help='path to config file with all params')

	args = parser.parse_args()
	print(args)

	# sanity checks and setting up things

	# make sure input files exist and output doesn't
	assert Path(args.input_image_dir).is_dir()
	assert Path(args.config_file).is_file()
	assert not Path(args.output_dir).exists()

	# create output dir
	Path(args.output_dir).mkdir(parents=True)

	# copy input image and config file to output dir
	Path(os.path.join(args.output_dir, "img")).mkdir()
	Path(os.path.join(args.output_dir, "config")).mkdir()
	Path(os.path.join(args.output_dir, "ckpt")).mkdir()

	shutil.copytree(args.input_image_dir, os.path.join(args.output_dir, "img/train_set"))
	shutil.copy(args.config_file, os.path.join(args.output_dir, "config"))

	# read the input json config
	with open(args.config_file) as f:
		config = json5.load(f)
	print(config)

	# call the training routine
	train(args.input_image_dir, args.output_dir, config)
