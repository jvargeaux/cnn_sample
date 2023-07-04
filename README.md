
# Digit Classifier

This is a Hello World style sample project I made for learning Convolutional Neural Networks (CNNs) in PyTorch. The project is a Flask app that takes in a .png image and produces a numerical digit [0-9] from a model trained in PyTorch on the MNIST handwritten digit dataset.


## API

`[POST] /api/digit`

**Request**
>`file: image/png`

The user can either upload a .png file or draw on the HTML5 canvas, which is converted to png and uploaded.

**Response**
>`prediction: (digit)`
>`percentages: [10]`

The model will respond with a 10-element array of floating point numbers containing the probabilities of each digit.


## Network Parameters

Network type: Convolutional Neural Network
Loss criterion: Cross-Entropy Loss
Optimizer: Adam, extension of Stochastic Gradient Descent (SGD)
Learning Rate: 0.01


## Layer Architecture

**Input Layer**
>Takes transformed and normalized .png image input in the shape of [n, 1, 28, 28].

**Convolution Layers**
>Uses 2 convolution layers, with activation functions and max pooling.  
Conv2d -> ReLU -> MaxPool2d

**Flatten Image Data**
>Refits the image data into 1 dimensional data for the linear layers.

**Fully Connected Layers**
>Uses 3 linear layers, with activation functions.  
Linear -> ReLU

**Output Layer**
>Outputs the final layer of 10 class options (digits 0-9), with Softmaxing done by the optimizer in training or manually in evaluation mode.


## Project Structure

/
> Contains the model state .pth file used in the app, and top-level project resources like .env and resources.txt

/app
> Contains Flask app code, including templates and static frontend resources

/training
> Contains the training script, model class, and downloaded training data