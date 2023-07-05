
# Digit Classifier

This is a Hello World style sample project I made for learning Convolutional Neural Networks (CNNs) in PyTorch. The project is a Flask app that takes in a .png image and produces a numerical digit [0-9] from a model trained in PyTorch on the MNIST handwritten digit dataset.

Project Demo URL: http://167.172.139.55/cnn/


# Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [API](#api)
- [Project Architecture](#project-architecture)
  - [Network Parameters](#network-parameters)
  - [Layer Architecture](#layer-architecture)


# Getting Started

## Prerequisites

To run the application locally, you will need the following prerequisites, found in [requirements.txt](./requirements.txt).

- Python 3.10 or later
- PyTorch
- Torchvision
- Pillow
- Matplotlib
- Flask
- python-dotenv

It is recommended to use a package manager such as conda or pip and create a virtual environment with the specific versions of each dependency.

- conda

> Go to [Anaconda's website](https://www.anaconda.com/download) and download the version for your OS

or

- pip and virtualenv

> pip is installed automatically with Python if downloaded from the [Python website](https://www.python.org/)

Linux / MacOS

```
python3 -m pip install --user virtualenv
```

Windows

```
py -m pip install --user virtualenv
```


## Installation

With the above installed locally on your machine, follow the process below.

---

1. Clone the repo

```
git clone https://github.com/jvargeaux/cnn_sample.git
```

---

2. cd into the project folder

```
cd mydir/cnn_sample/
```

---

3. Create a virtual environment with the dependencies listed in requirements.txt, and activate the virtual environment

>conda

```
conda create -n cnn_sample_env --file requirements.txt
conda activate cnn_sample_env
```

>pip

Linux / MacOS

```
python3 -m venv cnn_sample_env
source cnn_sample_env/bin/activate
python3 -m pip install -r requirements.txt
```

Windows

```
py -m venv cnn_sample_env
.\env\Scripts\activate
py -m pip install -r requirements.txt
```

---

4. Set the flask environment variables using one of the following methods

>Create a .env file in the project root directory with the following variables

```
FLASK_APP=app/main.py
FLASK_DEBUG=True
```

>Set the environment variables from the command line

Linux / MacOS

```
export FLASK_APP=app/main.py
export FLASK_DEBUG=True
```

Windows

```
set FLASK_APP=app/main.py
set FLASK_DEBUG=True
```

---

5. Run the flask app, which will open the app on host 127.0.0.1, port 5000

```
flask run
```


# Usage

## Training

To train the model, run the following command from the project root directory.

Linux / MacOS

```
python training/train.py
```

Windows

```
py training/train.py
```

---

### Data

The MNIST dataset is a famous set of 60,000 samples of hand-drawn black and white images. We slice these up into batches and train them in steps within each epoch.

![MNIST dataset](https://datasets.activeloop.ai/wp-content/uploads/2019/12/MNIST-handwritten-digits-dataset-visualized-by-Activeloop.webp)

---

### Examples

To show examples of the dataset in the terminal, simply uncomment the second line in the main() function, found on line 103.

```python
# trainer.display_example_data()
```

---

### Save model

To save the model state to a .pth file, uncommet the line in the main() function, found on line 106.

```python
# trainer.save_model()
```

---

### Modify hyper parameters

To modify the hyper parameters, navigate to [training/train.py](./training/train.py) and change the number of epochs and batch size in the __init__() function.

```python
# Set hyper parameters
self.num_epochs = 4
self.batch_size = 60 # split data into batches for training
```

The loss criterion, learning rate, and optimizer can be found in the train() function.

```python
# Define loss criterion
criterion = nn.CrossEntropyLoss() # cross-entropy loss for multi-class classification problem

# Define learning rate & optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) # extension of Stochastic Gradient Descent (SGD)
```


## API

`[POST] /api/digit`

**Request**
>`file: image/png`

The user can either upload a .png file or draw on the HTML5 canvas, which is converted to png and uploaded.

**Response**
>`prediction: (digit)`
>`percentages: [10]`

The model will respond with a 10-element array of floating point numbers containing the probabilities of each digit.


# Project Architecture

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