# ANN-From-Scratch
Implementation of a multi layer perceptron artificial neural network from scratch that is tested using the MNIST dataset.

## Prerequisites

The [NumPy](https://github.com/numpy/numpy) module is used for numerical vector and matrix calculations:

```bash
pip install numpy
```

The [Keras](https://github.com/keras-team/keras) module is used to import the MNIST dataset:

```bash
pip install keras
```

The [tqdm](https://github.com/tqdm/tqdm) module is used for creating the progress bar:

```bash
pip install tqdm
```

## Design choices 

The neural network implementation uses the following design choices:

* Sigmoid activation function.

* Xavier Glorot initialization of the weights.

* Zero initialization of the biases.

* Mean squared error cost function.

## Usage

The neural network implementation, can be found in the `ann.py` module where the `NeuralNetwork` class exists.

To test the neural network on the MNIST dataset, run the `mnist.py` file. This script will train and test a neural network with the given parameters.

## Results

The neural network reaches an accuracy of about 90.5 % on MNIST dataset with the parameters used in the `mnist.py` file.
