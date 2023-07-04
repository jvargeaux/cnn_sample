import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from matplotlib import pyplot



# Set device, use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


# Set hyper parameters
num_epochs = 2
batch_size = 60 # split data into batches for training
num_classes = 10 # num possible digits
# Probabilities, e.g.
# If 0: [1, 0, 0, ...]
# If 1: [0, 1, 0, ...]
# If 2: [0, 0, 1, ...]



# PREPARE DATA

# Download MNIST (digits) dataset
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Load data into batches for training
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# # Show shape and 6 example data sample images
# examples = iter(train_loader)
# samples, labels = examples.next()
# print(f'Samples: {samples.shape}')
# print(f'Labels: {labels.shape}')
# for i in range(6):
#     pyplot.subplot(2, 3, i+1)
#     pyplot.imshow(samples[i][0], cmap='gray')
# pyplot.show()



# CREATE CNN MODEL

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
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
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


model = DigitClassifierCNN().to(device)

# Define loss criterion
criterion = nn.CrossEntropyLoss() # cross-entropy loss for multi-class classification problem

# Define learning rate & optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # extension of Stochastic Gradient Descent (SGD)



# TRAINING

num_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [60, 1, 28, 28]

        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{num_steps}], Loss: {loss.item():.4f}')

print('Finished Training!')



# TESTING

with torch.no_grad(): # Don't use gradients!
    num_correct = 0
    num_samples = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1) # (value, index)
        num_samples += labels.shape[0]
        num_correct += (predicted == labels).sum().item()

    accuracy = 100.0 * num_correct / num_samples
    print(f'Accuracy of the network: {accuracy}%')