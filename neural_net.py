try:
    # Try to import CuPy
    import cupy as cp
    # Attempt to allocate memory on a GPU to confirm its presence
    cp.array([1])
    # If successful, alias CuPy as np to use it as if it were NumPy
    np = cp
except (ImportError):
    # If CuPy is not installed, fall back to NumPy
    import numpy as np
except (cp.cuda.runtime.CUDARuntimeError):
    # If no GPU is found, fall back to NumPy
    import numpy as np


from layers import Layer
from activation import Activation
from typing import Tuple, Type

class NeuralNet:
    """
    Represents a neural network.

    Args:
        input_size (int): The size of the input layer.
        *layers (Tuple[int, Type[Layer]]): Variable-length argument list of tuples, where each tuple contains the size of a layer and the type of layer.

    Attributes:
        input_size (int): The size of the input layer.
        layers (list): A list of layer objects that make up the neural network.
        output_size (int): The size of the output layer.
    """

    def __init__(self, input_size, *layers: Tuple[int, Type[Layer], Activation]):
        self.input_size = input_size
        #input size of a layer is the output size of the previous layer
        layer_sizes = [input_size] + [layer[0] for layer in layers]
        self.layers = [layer[1](layer_sizes[i], layer_sizes[i+1], layer[2]) for i, layer in enumerate(layers)]
        self.output_size = layer_sizes[-1]

    def initialize(self, batch_size=None):
        """
        Initializes the neural network by calling the initialize method of each layer.

        Args:
            batch_size (int, optional): The batch size for the input data. Defaults to None.
        """
        for layer in self.layers:
            layer.initialize(batch_size)

    def forward(self, inputs):
        """
        Performs forward propagation through the neural network.

        Args:
            inputs: The input data to be propagated through the network.

        Returns:
            The output of the neural network after forward propagation.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def batch_forward(self, inputs):
        """
        Perform a forward pass through the neural network for a batch of inputs.

        Args:
            inputs: The input data for the batch.

        Returns:
            The output of the neural network after the forward pass.
        """
        for layer in self.layers:
            inputs = layer.batch_forward(inputs)
        return inputs
    
    def forward_n_times(self, inputs, n):
        """
        Perform forward propagation through the neural network n times.

        Args:
            inputs: The input data to be propagated through the network.
            n (int): The number of times to perform forward propagation.

        Returns:
            The output of the neural network for every forward pass.
        """
        outputs = []
        for i in range(n):
            output = self.forward(inputs[i])
            outputs.append(output)
        return outputs
    
    def batch_forward_n_times(self, inputs, n):
        """
        Perform forward propagation through the neural network n times for a batch of inputs.

        Args:
            inputs: The input data for the batch.
            n (int): The number of times to perform forward propagation.

        Returns:
            The output of the neural network for every forward pass.
        """
        outputs = []
        for i in range(n):
            output = self.batch_forward(inputs[i])
            outputs.append(output)
        return outputs
    
    def mutate(self, weight_mutation_rate=0.1, weight_mutation_strength=0.1, bias_mutation_rate=0.1, bias_mutation_strength=0.1, *, cooldown_mutation_rate=0.1, cooldown_mutation_strength=1):
        """
        Mutates the neural network by applying random mutations to the weights and biases of its layers.

        Parameters:
        - weight_mutation_rate (float): The probability of mutating each weight value.
        - weight_mutation_strength (float): The maximum magnitude of the weight mutation.
        - bias_mutation_rate (float): The probability of mutating each bias value.
        - bias_mutation_strength (float): The maximum magnitude of the bias mutation.
        - cooldown_mutation_rate (float): The probability of mutating the cooldown valus (applicable only to RGLayer).
        - cooldown_mutation_strength (float): The maximum magnitude of the cooldown mutations (applicable only to RGLayer).
        """
        for layer in self.layers:
            if layer.__class__.__name__ == 'RGLayer':
                layer.mutate(weight_mutation_rate, weight_mutation_strength, bias_mutation_rate, bias_mutation_strength, cooldown_mutation_rate, cooldown_mutation_strength)
            else:
                layer.mutate(weight_mutation_rate, weight_mutation_strength, bias_mutation_rate, bias_mutation_strength)

    def crossover(self, other):
        """
        Creates a new neural network by performing crossover with another neural network.

        Args:
            other (NeuralNet): The other neural network to perform crossover with.

        Returns:
            NeuralNet: The new neural network created by performing crossover with the other neural network.
        """
        new_net = NeuralNet(self.input_size, *[(layer.output_size, layer.__class__) for layer in self.layers])
        for i, layer in enumerate(new_net.layers):
            layer.crossover(other.layers[i])
        return new_net
