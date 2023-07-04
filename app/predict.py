import sys
sys.path.append('../training')

import io
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from training.model import DigitClassifierCNN


# Load the model state into a new instance
model = DigitClassifierCNN()
PATH = ''
model.load_state_dict(torch.load(PATH))
model.eval()


# Refit image data to be read by the model
def transform_image(image_bytes):
  transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,)), # from MNIST
  ])
  image = Image.open(io.BytesIO(image_bytes))

  return transform(image).unsqueeze(0)


# Use our trained model on the image data we receive
def get_prediction(image_tensor):
  images = image_tensor.reshape(-1, 28*28)
  outputs = model(images)
  value, predicted = torch.max(outputs.data, 1)
  return predicted