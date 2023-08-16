# this file contains simple utility functions

from PIL import Image
from torchvision import transforms

def read_image_tensor(image_path, image_size=(512, 512)):
	"""
	given an image path
	read the image
	and return as a tensor
	tensor is [-1.0, 1.0] normalized
	"""

	img = Image.open(image_path)
	img_transforms = transforms.Compose([
			transforms.Resize(image_size),
			transforms.ToTensor()
		])

	return 2 * img_transforms(img) - 1

def save_image_tensor(image_tensor, output_filename):
	"""
	given a tensor image, display it
	just rescale to [0, 1] before displaying
	"""

	min_val = image_tensor.min()
	max_val = image_tensor.max()

	image_tensor = (image_tensor - min_val) / (max_val - min_val)

	img_transform = transforms.ToPILImage()
	img_pil = img_transform(image_tensor)

	# img_pil.show()
	# display(img_pil)

	img_pil.save(output_filename)

	return
