import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from matplotlib import pyplot
from model import DigitClassifierCNN



class DigitClassifierCNNTrainer():
  def __init__(self):
    # Set device, use GPU if available
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {self.device}')

    # Set hyper parameters
    self.num_epochs = 4
    self.batch_size = 60 # split data into batches for training

    # Download MNIST (digits) dataset
    self.train_data = torchvision.datasets.MNIST(root='./training/data', train=True, download=True, transform=transforms.ToTensor())
    self.test_data = torchvision.datasets.MNIST(root='./training/data', train=False, download=True, transform=transforms.ToTensor())

    # Load data into batches for training
    self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)
    self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False)

    # Initialize model
    self.model = DigitClassifierCNN().to(self.device)


  def display_example_data(self):
    examples = iter(self.train_loader)
    samples, labels = examples.next()
    print(f'Samples: {samples.shape}')
    print(f'Labels: {labels.shape}')
    for i in range(6):
        pyplot.subplot(2, 3, i+1)
        pyplot.imshow(samples[i][0], cmap='gray')
    pyplot.show()


  def train(self):
    # Define loss criterion
    criterion = nn.CrossEntropyLoss() # cross-entropy loss for multi-class classification problem

    # Define learning rate & optimizer
    learning_rate = 0.01
    optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) # extension of Stochastic Gradient Descent (SGD)

    num_steps = len(self.train_loader)

    for epoch in range(self.num_epochs):
      for i, (images, labels) in enumerate(self.train_loader):
        # origin shape: [60, 1, 28, 28]
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Forward Pass
        outputs = self.model(images)
        loss = criterion(outputs, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
          print(f'Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{num_steps}], Loss: {loss.item():.4f}')

    print('Finished Training!')


  def test_model(self):
    with torch.no_grad(): # Don't use gradients!
      num_correct = 0
      num_samples = 0

      for images, labels in self.test_loader:
        images = images.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(images)

        _, predicted = torch.max(outputs, 1) # (value, index)
        num_samples += labels.shape[0]
        num_correct += (predicted == labels).sum().item()

      accuracy = 100.0 * num_correct / num_samples
      print(f'Accuracy of the network: {accuracy}%')


  def save_model(self):
    PATH = 'model.pth'
    torch.save(self.model.state_dict(), PATH)



# Main if run as script

def main():
  trainer = DigitClassifierCNNTrainer()
  # trainer.display_example_data()
  trainer.train()
  trainer.test_model()
  # trainer.save_model()

if __name__ == '__main__':
  main()