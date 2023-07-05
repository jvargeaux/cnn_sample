import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.append(parent_dir)
  
import io
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from training.model import DigitClassifierCNN



# Load the model state into a new instance
class DigitClassifierCNNEvaluator():
  def __init__(self):
    self.model = DigitClassifierCNN()

    # Load model state from .pth file
    PATH = parent_dir + '/model.pth'
    self.model.load_state_dict(torch.load(PATH))
    self.model.eval()


  # Refit image data to be read by the model
  def transform_image(self, image_bytes):
    transform = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.Resize((28, 28)),
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.1307,), std=(0.3081,)), # from MNIST
    ])
    image = Image.open(io.BytesIO(image_bytes))

    return transform(image).unsqueeze(0)


  # Use our trained model on the image data we receive
  def predict(self, image_tensor):
    outputs = self.model(image_tensor)
    
    # Use Softmax to get percentages for all classes
    all_outputs = torch.softmax(outputs.data, dim=1)
    all_outputs = all_outputs.numpy()[0]
    percentages = all_outputs.tolist()

    _, predicted = torch.max(outputs.data, 1)
    return predicted, percentages