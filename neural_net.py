# try:
#     # Try to import CuPy
#     import cupy as cp
#     # Attempt to allocate memory on a GPU to confirm its presence
#     cp.array([1])
#     # If successful, alias CuPy as np to use it as if it were NumPy
#     np = cp
# except (ImportError):
#     # If CuPy is not installed, fall back to NumPy
#     import numpy as np
# except (cp.cuda.runtime.CUDARuntimeError):
#     # If no GPU is found, fall back to NumPy
import numpy as np
import matplotlib.pyplot as plt
import pickle

from layers import Layer
from activation import Activation
from typing import Tuple, Type, Optional, Unpack

class NeuralNet:
    """
    Represents a neural network.

    Args:
        input_size (int): The size of the input layer.
        *layers (Tuple[int, Type[Layer], Activation]): The layers of the neural network.

    Attributes:
        input_size (int): The size of the input layer.
        layers (List[Layer]): The layers of the neural network.
        output_size (int): The size of the output layer.

    """

    def __init__(self, input_size: int, *layers: Tuple[int, Type[Layer], Activation]) -> None:
        self.input_size = input_size
        #input size of a layer is the output size of the previous layer
        layer_sizes = [input_size] + [layer[0] for layer in layers]
        self.layers = [layer[1](layer_sizes[i], layer_sizes[i+1], layer[2]) for i, layer in enumerate(layers)]
        self.output_size = layer_sizes[-1]

    def copy(self) -> 'NeuralNet':
        """
        Creates a copy of the neural network.

        Returns:
            NeuralNet: A copy of the neural network.
        """
        new_net = NeuralNet(self.input_size, *[(layer.neuron_count, layer.__class__, layer.activation_function) for layer in self.layers])
        for i, layer in enumerate(new_net.layers):
            layer = self.layers[i].copy()
        return new_net
    
    def copy_structure(self) -> 'NeuralNet':
        """
        Creates a copy of the neural network with the same structure but with uninitialized weights and biases.

        Returns:
            NeuralNet: A copy of the neural network with the same structure but with uninitialized weights and biases.
        """
        new_net = NeuralNet(self.input_size, *[(layer.neuron_count, layer.__class__, layer.activation_function) for layer in self.layers])
        return new_net

    def initialize(self, batch_size: Optional[int] = None) -> None:
        """
        Initializes the neural network by calling the initialize method of each layer.

        Args:
            batch_size (int, optional): The batch size for the input data. Defaults to None.
        """
        for layer in self.layers:
            layer.initialize(batch_size)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
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
    
    def batch_forward(self, inputs: np.ndarray) -> np.ndarray:
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
    
    def forward_n_times(self, inputs: np.ndarray, n: int) -> list[np.ndarray]:
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
    
    def batch_forward_n_times(self, inputs: np.ndarray, n: int) -> list[np.ndarray]:
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
    
    def mutate(self, weight_mutation_rate: float = 0.1, weight_mutation_strength: float = 0.1, bias_mutation_rate: float = 0.1, bias_mutation_strength: float = 0.1, **mutations: float | int) -> None:
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
                cooldown_mutation_rate: float = mutations.get("cooldown_mutation_rate", 0.1)
                cooldown_mutation_strength: int = int(mutations.get("cooldown_mutation_strength", 1))
                layer.mutate(weight_mutation_rate, weight_mutation_strength, bias_mutation_rate, bias_mutation_strength, cooldown_mutation_rate=cooldown_mutation_rate, cooldown_mutation_strength=cooldown_mutation_strength)
            else:
                layer.mutate(weight_mutation_rate, weight_mutation_strength, bias_mutation_rate, bias_mutation_strength)

    def crossover(self, other: 'NeuralNet') -> 'NeuralNet':
        """
        Creates a new neural network by performing crossover with another neural network.

        Args:
            other (NeuralNet): The other neural network to perform crossover with.

        Returns:
            NeuralNet: The new neural network created by performing crossover with the other neural network.
        """
        new_net = NeuralNet(self.input_size, *[(layer.neuron_count, layer.__class__, layer.activation_function) for layer in self.layers])
        for i, layer in enumerate(new_net.layers):
            layer.crossover(other.layers[i])
        return new_net
    
    def visualize_weights_bias(self) -> None:
        """
        Visualizes the weights and biases of the neural network.
        """
        biases = [layer.get_biases() for layer in self.layers]
        weights = [layer.get_weights(True) for layer in self.layers]
        max_weight_count = max([len(layer_weights) for layer_weights in weights])
        other = [layer.get_other(True) for layer in self.layers]
        max_other_count = max([len(layer_other) for layer_other in other])
        print(max_other_count)

        # use matplotlib to visualize the weights and biases. First row is biases, next rows are weights, last rows are other
        # the number of columns is the number of layers
        fig, axs = plt.subplots(1 + max_weight_count + max_other_count, len(self.layers), figsize=(5*len(self.layers), 2*(1 + max_weight_count + max_other_count)))
        fig.suptitle('Weights and Biases')

        # display the weights and biases as heatmaps

        for i, layer_biases in enumerate(biases):
            axs[0, i].imshow(layer_biases.reshape(1, -1), cmap='coolwarm', aspect='equal')
            axs[0, i].set_title('Biases')
            axs[0, i].set_yticks([])
            axs[0, i].set_xticks([])

        # weights is a list of dictionaries, each dictionary contains the weights for a layer
        for i, layer_weights in enumerate(weights):
            for j, (name, weight) in enumerate(layer_weights.items()):
                axs[j+1, i].imshow(weight, cmap='coolwarm', aspect='equal')
                axs[j+1, i].set_title(name)
                axs[j+1, i].set_yticks([])
                axs[j+1, i].set_xticks([])
            # hide the remaining subplots
            for j in range(len(layer_weights), max_weight_count):
                axs[j+1, i].axis('off')

        # other is a list of dictionaries, each dictionary contains the other parameters for a layer
        for i, layer_other in enumerate(other):
            for j, (name, other_value) in enumerate(layer_other.items()):
                if len(other_value.shape) == 1:
                    other_value = other_value.reshape(1, -1)
                axs[j+1+max_weight_count, i].imshow(other_value, cmap='coolwarm', aspect='equal')
                axs[j+1+max_weight_count, i].set_title(name)
                axs[j+1+max_weight_count, i].set_yticks([])
                axs[j+1+max_weight_count, i].set_xticks([])
            # hide the remaining subplots
            for j in range(len(layer_other), max_other_count):
                axs[j+1+max_weight_count, i].axis('off')


        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # add layer labels to each column
        for i in range(len(self.layers)):
            if i != len(self.layers) - 1:
                fig.text((i+0.5)/len(self.layers), 0.9, f'Layer {i+1}', ha='center', va='center', fontsize=20)
            else:
                fig.text((i+0.5)/len(self.layers), 0.9, f'Output Layer', ha='center', va='center', fontsize=20)

        plt.show()

    def save(self, name: str, file_path: str, *, replace: bool = False) -> None:
        """
        Save the neural network to a file.

        Args:
            name (str): The name of the file.
            file_path (str): The path to the file.
        """
        try:
            with open(f"{file_path}/{name}.nn", 'xb') as file:
                pickle.dump(self, file)
        except FileExistsError:
            if replace:
                with open(f"{file_path}/{name}.nn", 'wb') as file:
                    pickle.dump(self, file)
            else:
                raise FileExistsError("File already exists. Set replace to True to overwrite the file.")

    @staticmethod
    def load(file_path: str) -> 'NeuralNet':
        """
        Load a neural network from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            NeuralNet: The neural network loaded from the file.
        """
        # make sure the file has the correct extension
        if not file_path.endswith('.nn'):
            raise ValueError("File must have the extension '.nn'")

        with open(file_path, 'rb') as file:
            return pickle.load(file)