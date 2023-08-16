# this file contains the generation routine

from argparse import ArgumentParser
import json5
import os
from pathlib import Path
from tqdm import tqdm

from diffusion import *
from utils import *

def generate(model_file, output_dir, config, num_samples):
	# the main wrapper function around generation
	
	# get the device to be used
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print(device)

	# load the model
	denoise_model = torch.load(model_file).to(device)

	# iterate over num_samples
	for idx in tqdm(range(num_samples), desc='Generating'):

		# generate image path for this timestep assuming 3 channels
		x_T = torch.normal(mean=torch.zeros([3] + config["image_size"])).to(device).unsqueeze(0)
		
		denoised_path = reverse_diffusion_linear_schedule(denoise_model, x_T, config["beta_1"], 
									config["beta_T"], config["num_diffusion_steps"], device)
		save_image_tensor(denoised_path[config["num_diffusion_steps"]][0], os.path.join(output_dir, f"sample_{idx}.jpg"))
		
		# denoise_model.train()
		del denoised_path

if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('--model_file', type=str, help='path to the .pt model file')
	parser.add_argument('--output_dir', type=str, help='path to output directory')
	parser.add_argument('--config_file', type=str, help='path to config file with all params')
	parser.add_argument('--num_samples', type=int, help='number of samples to be taken from the model')

	args = parser.parse_args()
	print(args)

	# sanity checks and setting up things

	# make sure input files exist and output doesn't
	assert Path(args.model_file).is_file()
	assert Path(args.config_file).is_file()
	assert not Path(args.output_dir).exists()

	# create output dir
	Path(args.output_dir).mkdir(parents=True, exist_ok=False)

	# read the input json config
	with open(args.config_file) as f:
		config = json5.load(f)
	print(config)

	# call the generation routine
	generate(args.model_file, args.output_dir, config, args.num_samples)
