## About

This is a very simple, bare-bones implementation of Diffusion from Scratch. The objective is also as simple as it gets: train the model on a single image. I am using a very simple unet model (I use the `denoising_diffusion_pytorch` library here). Rest all diffusion related functions are written from scratch. I have referred to a fewe very good sources which I have mentioned below.

Update 1: The earlier plan was to train on a single image; that was successful. Now, the goal is to train on a dataset.

However, I'm just being very lazy with it. It only supports a really small ideally maybe a dozen or so images. Put them in a batch and iterate.

## Usage

```bash

# train
python train_diffusion_model.py --input_image_dir [path_to_input_image_dir] --output_dir [path_to_output_dir] --config_file [path_to_config_file]

# generate
python generate_from_diffusion_model.py --model_file [path_to_model_file] --output_dir [path_to_output_dir] --config_file [path_to_config_file] --num_samples [number_of_samples]

```

## Results

For validating the correctness, I trained the model on 4 images and here are how the results look.

### Training Data

<p float="left">
  <img src="/img/beagle.jpg" width="256"/>
  <img src="/img/dalmation.jpg" width="256"/>
  <img src="/img/golden_retriever.jpg" width="256"/>
  <img src="/img/pembroke-welsh-corgi.jpg" width="256"/>
</p>

### Sampled Images

<p float="left">
  <img src="/img/beagle.jpg" width="256"/>
  <img src="/img/dalmation.jpg" width="256"/>
  <img src="/img/golden_retriever.jpg" width="256"/>
  <img src="/img/pembroke-welsh-corgi.jpg" width="256"/>
</p>

## Resources

- https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/
- https://huggingface.co/blog/annotated-diffusion