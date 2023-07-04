import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from matplotlib import pyplot
from model import DigitClassifierCNN



# Set device, use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


# Set hyper parameters
num_epochs = 4
batch_size = 60 # split data into batches for training


# PREPARE DATA

# Download MNIST (digits) dataset
train_data = torchvision.datasets.MNIST(root='./training/data', train=True, download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='./training/data', train=False, download=True, transform=transforms.ToTensor())

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



# CREATE CNN MODEL, LOSS, OPTIMIZER

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



# SAVE MODEL

PATH = 'model.pth'
torch.save(model.state_dict(), PATH)