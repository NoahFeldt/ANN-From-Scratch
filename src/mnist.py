import keras
import numpy as np
from tqdm import tqdm
import pickle

import ann

# Set seed for random number generation
np.random.seed(0)

def flatten(x):
    """Flattens images into a 1D array.
    """
    
    x_flat = []

    for i in range(0, len(x)):
        x_flat.append(x[i].flatten())

    return x_flat

def one_hot(y):
    """Converts array of numbers into an array of one hot lists.
    """

    y_one_hot = []

    for i in range(0, len(y)):
        one_hot = [0] * 10
        one_hot[y[i]] = 1
        one_hot = np.array(one_hot)

        y_one_hot.append(one_hot)

    return y_one_hot

def train(nn, x, y, learning_rate):
    """Train neural network on data.
    """
    
    for i in tqdm(range(0, len(x))):
        nn.backPropagation(x[i], y[i], learning_rate)

    return nn

def test_nn(nn, x, y):
    """Test neural network on data.
    """
    
    correct = 0
    total = 0

    for i in range(0, len(x)):
        output = nn.feedForward(x[i])

        max_index = output.argmax()

        if max_index == y[i]:
            correct += 1

        total += 1

    print(f"Accuracy: {correct / total * 100} %")

def save_nn(save_path, nn):
    """Save neural network object to pickle file.
    """

    with open(save_path, "wb") as file:
        pickle.dump(nn, file)

def load_nn(save_path):
    """Save neural network object to pickle file.
    """

    with open(save_path, "rb") as file:
        return pickle.load(file)

def main():
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float64) / 255.0
    x_test = x_test.astype(np.float64) / 255.0

    # Flatten images
    x_train_flat = flatten(x_train)
    x_test_flat = flatten(x_test)

    # Create one hot data
    y_train_one_hot = one_hot(y_train)
    y_test_one_hot = one_hot(y_test)

    layers = [784, 64, 10]
    learning_rate = 0.01
    epochs = 3

    nn = ann.NeuralNetwork(layers)

    for epoch in range(0, epochs):
        print(f"Training on epoch: {epoch}")

        # Training
        nn = train(nn, x_train_flat, y_train_one_hot, learning_rate)

        # Testing
        test_nn(nn, x_test_flat, y_test)

    # Save neural network to pickle file
    save_path = "ann.pickle"
    save_nn(save_path, nn)

if __name__ == "__main__":
    main()
