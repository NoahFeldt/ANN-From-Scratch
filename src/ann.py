import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        """NeuralNetwork constructor
        
        Initialize NeuralNetwork class based on layers information.
        
        Args:
            layers: A list object that contains the number of neurons in each layer from input to output.
        
        Returns:
            A NeuralNetwork object that contains weights and bias information.
        """

        # Initialize layers as a list that contains the number of neurons in each layer
        self.layers = layers

        # Initialize biases
        self.biases = []

        for i in range(1, len(layers)):
            # Zero initialization
            bias_layer = np.zeros(layers[i])

            self.biases.append(bias_layer)

        # Initialize weights
        self.weights = []

        for i in range(1, len(layers)):
            # Xavier Glorot initialization 
            n_in = layers[i - 1]
            n_out = layers[i]

            limit = np.sqrt(6 / (n_in + n_out))
            weights_matrix = np.random.uniform(-limit, limit, size=(n_out, n_in))

            self.weights.append(weights_matrix)

    def sigmoid(self, x):
        """Sigmoid function.
        """

        # Returns 1 / (1 + e^-x)
        return 1 / (1 + np.exp(-x)) 

    def sigmoidPrime(self, x):
        """The derivative of the sigmoid function.
        """

        # Calculate sigmoid of x
        sigmoid_x = self.sigmoid(x)

        # Returns sigmoid(x) * (1 - sigmoid(x))
        return sigmoid_x * (1 - sigmoid_x)

    def feedForward(self, input) -> np.ndarray:
        """Feed Forward.
        """

        # Initialize neurons before activation (weighted sum)
        self.neurons = [input]

        # Initialize neurons after activation
        self.activations = [input]

        for i in range(0, len(self.layers) - 1):
            # Calculate value of neurons in the next layer as a matrix multiplication of weights and neurons in the previous layer + biases
            neurons_layer = np.matmul(self.weights[i], self.neurons[i]) + self.biases[i]
            # Add list of neurons in layer to neurons list
            self.neurons.append(neurons_layer)

            # Calculate neuron values after activation
            activations_layer = self.sigmoid(neurons_layer)
            # Add list of activated neurons to activations list
            self.activations.append(activations_layer)

        # Return last activation layer as output of the neural network
        return self.activations[len(self.activations) - 1]

    def backPropagation(self, input, target, learning_rate):
        """Back Propagation.
        """

        self.feedForward(input)

        # Derivative of cost function with respect to last layer
        c_a = 2 * (self.activations[len(self.activations) - 1] - target)

        # Derivative of activation with respect to pre activation
        a_z = self.sigmoidPrime(self.neurons[len(self.neurons) - 1])

        # Derivative of pre activation with respect to weight
        z_w = self.activations[len(self.activations) - 2]

        c_a = np.array([c_a])
        a_z = np.array([a_z])
        z_w = np.array([z_w])

        # Adjust last layer weights
        weights_gradient = np.matmul((c_a * a_z).transpose(), z_w)
        self.weights[len(self.weights) - 1] -= weights_gradient * learning_rate

        # Adjust last layer biases
        bias_gradient = c_a * a_z
        self.biases[len(self.biases) - 1] -= bias_gradient[0] * learning_rate

        # Adjust weights and biases for hidden layers
        for i in range(0, len(self.layers) - 2):
            # Calculate gradient of second last layer neurons
            neurons_gradient = np.matmul((c_a * a_z), self.weights[len(self.weights) - 1 - i])

            # derivative cost function with respect to last layer
            c_a = neurons_gradient[0]

            # derivative activation with respect to pre activation
            a_z = self.sigmoidPrime(self.neurons[len(self.neurons) - 2 - i])

            # derivative pre activation with respect to weight
            z_w = self.activations[len(self.activations) - 3 - i]

            c_a = np.array([c_a])
            a_z = np.array([a_z])
            z_w = np.array([z_w])

            weights_gradient = np.matmul((c_a * a_z).transpose(), z_w)

            self.weights[len(self.weights) - 2 - i] -= weights_gradient * learning_rate

            # Adjust last layer biases
            bias_gradient = c_a * a_z

        self.biases[len(self.biases) - 2 - i] -= bias_gradient[0] * learning_rate
