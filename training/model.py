import torch
from torch import nn


# CONVOLUTION NEURAL NETWORK MODEL

# Probabilities, e.g.
# If 0: [1, 0, 0, ...]
# If 1: [0, 1, 0, ...]
# If 2: [0, 0, 1, ...]

class DigitClassifierCNN(nn.Module):

  # Define the layers in our network
  def __init__(self):
    super(DigitClassifierCNN, self).__init__()

    # Important!
    # Conv2d with 5x5 => -4 pixels on each dimension (e.g. 28x28 -> 24x24)
    # MaxPool2d with 2x2 => half size on each dimension (e.g. 28x28 -> 14x14)
    
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) # grayscale image = 1 input channel, 3x3 filter
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # input channels should match previous output channels
    self.relu = nn.ReLU() # Activation function, ReLU (Rectified Linear Unit) = negative numbers => 0
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(in_features=16*4*4, out_features=120) # Output after convolution layers: channels = 16, x = 4, y = 4
    self.fc2 = nn.Linear(in_features=120, out_features=84)
    self.fc3 = nn.Linear(in_features=84, out_features=10) # num possible digits
  
  # Proceed through each layer of the CNN
  def forward(self, x):
    # Convolution Layer 1
    out = self.conv1(x) # 28x28 -> 24x24
    out = self.relu(out)
    out = self.pool(out) # 24x24 -> 12x12

    # Convolution Layer 2
    out = self.conv2(out) # 12x12 -> 8x8
    out = self.relu(out)
    out = self.pool(out) # 8x8 -> 4x4

    # Flatten 2d image to 1 dimension
    out = out.view(-1, 16*4*4) # 16 channels, 4x4 image

    # Fully Connected Linear Layer 1
    out = self.fc1(out)
    out = self.relu(out)

    # Fully Connected Linear Layer 2
    out = self.fc2(out)
    out = self.relu(out)

    # Fully Connected Linear Layer 3
    out = self.fc3(out)
    # No Softmax or activation function on final layer, applied automatically in cross-entropy loss

    return out