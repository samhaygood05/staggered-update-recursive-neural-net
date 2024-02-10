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

class Layer:
    """
    Represents a layer in a neural network.

    Attributes:
        input_size (int): The size of the input to the layer.
        neuron_count (int): The number of neurons in the layer.
        activation_function (ActivationFunction): The activation function applied to the layer's output.
        w1 (numpy.ndarray): The weight matrix of shape (input_size, neuron_count).
        bias (numpy.ndarray): The bias vector of shape (neuron_count,).
    """

    def __init__(self, input_size, neuron_count, activation_function):
        self.input_size = input_size
        self.neuron_count = neuron_count
        self.activation_function = activation_function

        self.w1 = np.random.uniform(-1, 1, (input_size, neuron_count))
        self.bias = np.random.uniform(-1, 1, neuron_count)

    def copy(self):
        """
        Creates a copy of the layer.

        Returns:
            Layer: A new layer with the same attributes as the original.
        """
        new_layer = Layer(self.input_size, self.neuron_count, self.activation_function)
        new_layer.w1 = self.w1.copy()
        new_layer.bias = self.bias.copy()
        return new_layer

    def initialize(self, batch_size=None):
        """
        No initialization is needed for the base layer class. This method is here to be overridden by subclasses.
        """
        pass

    def forward(self, value):
        """
        Performs the forward pass of the layer.

        Args:
            value (numpy.ndarray): The input value to the layer.

        Returns:
            numpy.ndarray: The output value of the layer after applying the activation function.

        Raises:
            ValueError: If the input size does not match the layer's input size.
        """
        if len(value) != self.input_size:
            raise ValueError("Input size does not match layer input size")
        return self.activation_function.forward(np.dot(value, self.w1) + self.bias)

    def batch_forward(self, value):
        """
        Perform a forward pass on a batch of input values.

        Args:
            value (numpy.ndarray): The input values, with shape (batch_size, input_size).

        Returns:
            numpy.ndarray: The output values after applying the layer's activation function, with shape (batch_size, output_size).

        Raises:
            ValueError: If the input size does not match the layer's input size.
        """
        # if input is a 1d array, convert it to a 2d array
        if len(value.shape) == 1:
            value = value.reshape(1, -1)
        if value.shape[1] != self.input_size:
            raise ValueError("Input size does not match layer input size")
        return self.activation_function.batch_forward(np.dot(value, self.w1) + self.bias)

    def mutate(self, weight_mutation_rate=0.1, weight_mutation_strength=0.1, bias_mutation_rate=0.1, bias_mutation_strength=0.1):
        """
        Mutates the layer's weights and biases.

        Parameters:
            weight_mutation_rate (float): The probability of mutating the weights.
            weight_mutation_strength (float): The range of mutation for the weights.
            bias_mutation_rate (float): The probability of mutating the biases.
            bias_mutation_strength (float): The range of mutation for the biases.
        """
        if np.random.uniform(0, 1) < weight_mutation_rate:
            self.w1 += np.random.uniform(-weight_mutation_strength, weight_mutation_strength, (self.input_size, self.neuron_count))
        if np.random.uniform(0, 1) < bias_mutation_rate:
            self.bias += np.random.uniform(-bias_mutation_strength, bias_mutation_strength)

    def crossover(self, other: 'Layer'):
        """
        Crosses over the layer with another layer by randomly selecting the weights and biases from each of the layers.

        Args:
            other (Layer): The other layer to cross over with.

        Returns:
            Layer: A new layer that is the result of crossing over the two layers.
        """
        new_layer = Layer(self.input_size, self.neuron_count, self.activation_function)
        new_layer.w1 = np.where(np.random.randint(0, 2, (self.input_size, self.neuron_count)), self.w1, other.w1)
        new_layer.bias = np.where(np.random.randint(0, 2, self.neuron_count), self.bias, other.bias)
        return new_layer

class RGLayer(Layer):
    """
    Represents a layer in a neural network that uses a recursive graph structure.

    Attributes:
        input_size (int): The size of the input to the layer.
        neuron_count (int): The number of neurons in the layer.
        activation_function (ActivationFunction): The activation function applied to the layer's output.
        w1 (numpy.ndarray): The weight matrix of shape (input_size, neuron_count).
        w2 (numpy.ndarray): The weight matrix of shape (neuron_count, neuron_count).
        bias (numpy.ndarray): The bias vector of shape (neuron_count,).
        stored_values (numpy.ndarray): The stored values of the neurons.
        propagating_values (numpy.ndarray): The values propagating through the neurons.
    """
    def __init__(self, input_size, neuron_count, activation_function):
        super().__init__(input_size, neuron_count, activation_function)

        self.w2 = np.random.uniform(-1, 1, (neuron_count, neuron_count))
        self.stored_values = np.zeros(neuron_count)
        self.propagating_values = np.zeros(neuron_count)
        self.max_cooldowns = np.random.randint(0, 10, neuron_count)
        self.initial_cooldowns = np.random.randint(0, 10, neuron_count)
        self.current_cooldowns = self.initial_cooldowns.copy()

    def copy(self):
        """
        Creates a copy of the layer.

        Returns:
            RGLayer: A new layer with the same attributes as the original.
        """
        new_layer = RGLayer(self.input_size, self.neuron_count, self.activation_function)
        new_layer.w1 = self.w1.copy()
        new_layer.w2 = self.w2.copy()
        new_layer.bias = self.bias.copy()
        new_layer.stored_values = self.stored_values.copy()
        new_layer.propagating_values = self.propagating_values.copy()
        new_layer.max_cooldowns = self.max_cooldowns.copy()
        new_layer.initial_cooldowns = self.initial_cooldowns.copy()
        new_layer.current_cooldowns = self.current_cooldowns.copy()
        return new_layer

    def initialize(self, batch_size=None):
        """
        Initializes the neural network by setting the initial values for stored values, propagating values, and cooldowns.

        Parameters:
        - batch_size (int): The size of the batch. If None, the method initializes for a single instance. 
                            If specified, the method initializes for a batch of instances.

        Returns:
        None
        """
        if batch_size is None:
            self.stored_values = np.zeros(self.neuron_count)
            self.propagating_values = np.zeros(self.neuron_count)
        else:
            self.stored_values = np.zeros((batch_size, self.neuron_count))
            self.propagating_values = np.zeros((batch_size, self.neuron_count))
        self.current_cooldowns = self.initial_cooldowns.copy()

    def forward(self, value):
        """
        Performs the forward pass of the layer.

        Args:
            value (numpy.ndarray): The input value to the layer.

        Returns:
            numpy.ndarray: The output of the layer after applying the activation function.
        """
        # make sure value is the size of input_size
        if len(value) != self.input_size:
            raise ValueError("Input size does not match layer input size")
        # for neurons whose cooldown is 0, move the stored value to the propagating value
        self.propagating_values[self.current_cooldowns == 0] = self.stored_values[self.current_cooldowns == 0]
        # for neurons whose cooldown is 0, update the stored value
        self.stored_values[self.current_cooldowns == 0] = np.dot(value, self.w1[:, self.current_cooldowns == 0]) + np.dot(self.propagating_values, self.w2[:, self.current_cooldowns == 0]) + self.bias[self.current_cooldowns == 0]
        # decrement the cooldowns
        self.current_cooldowns -= 1
        # for neurons whose cooldown is -1, reset the cooldown to the max cooldown
        self.current_cooldowns[self.current_cooldowns == -1] = self.max_cooldowns[self.current_cooldowns == -1]
        # return the activation function applied to the propagating values
        return self.activation_function.forward(self.propagating_values)
    
    def batch_forward(self, value):
        """
        Perform a forward pass for a batch of input values through the layer.

        Args:
            value (numpy.ndarray): The input values for the layer.

        Returns:
            numpy.ndarray: The output values after applying the activation function.
        """
        # if input is a 1d array, convert it to a 2d array
        if len(value.shape) == 1:
            value = value.reshape(1, -1)
        # make sure value is the size of input_size
        if value.shape[1] != self.input_size:
            raise ValueError("Input size does not match layer input size")
        # for neurons whose cooldown is 0, move the stored value to the propagating value
        self.propagating_values[:, self.current_cooldowns == 0] = self.stored_values[:, self.current_cooldowns == 0]
        # for neurons whose cooldown is 0, update the stored value
        self.stored_values[:, self.current_cooldowns == 0] = np.dot(value, self.w1[:, self.current_cooldowns == 0]) + np.dot(self.propagating_values, self.w2[:, self.current_cooldowns == 0]) + self.bias[self.current_cooldowns == 0]
        # decrement the cooldowns
        self.current_cooldowns -= 1
        # for neurons whose cooldown is -1, reset the cooldown to the max cooldown
        self.current_cooldowns[self.current_cooldowns == -1] = self.max_cooldowns[self.current_cooldowns == -1]
        # return the activation function applied to the propagating values
        return self.activation_function.batch_forward(self.propagating_values)

    def mutate(self, weight_mutation_rate=0.1, weight_mutation_strength=0.1, bias_mutation_rate=0.1, bias_mutation_strength=0.1, cooldown_mutation_rate=0.1, cooldown_mutation_strength=1):
        """
        Mutates the layer's weights, biases, and cooldowns.

        Args:
            weight_mutation_rate (float): The probability of mutating the weights.
            weight_mutation_strength (float): The strength of the mutation for the weights.
            bias_mutation_rate (float): The probability of mutating the biases.
            bias_mutation_strength (float): The strength of the mutation for the biases.
            cooldown_mutation_rate (float): The probability of mutating the cooldowns.
            cooldown_mutation_strength (int): The strength of the mutation for the cooldowns.
        """
        super().mutate(weight_mutation_rate, weight_mutation_strength, bias_mutation_rate, bias_mutation_strength)
        
        if np.random.uniform(0, 1) < weight_mutation_rate:
            self.w2 += np.random.uniform(-weight_mutation_strength, weight_mutation_strength, (self.neuron_count, self.neuron_count))
        
        if np.random.uniform(0, 1) < cooldown_mutation_rate:
            self.initial_cooldowns += np.random.randint(-cooldown_mutation_strength, cooldown_mutation_strength, self.neuron_count)
        
        if np.random.uniform(0, 1) < cooldown_mutation_rate:
            self.max_cooldowns += np.random.randint(-cooldown_mutation_strength, cooldown_mutation_strength+1, self.neuron_count)

    def crossover(self, other: 'RGLayer'):
        """
        Crosses over the layer with another layer by randomly selecting the weights, biases, and cooldowns from each of the layers.

        Args:
            other (RGLayer): The other layer to cross over with.

        Returns:
            RGLayer: A new layer that is the result of crossing over the two layers.
        """
        new_layer = RGLayer(self.input_size, self.neuron_count, self.activation_function)
        new_layer.w1 = np.where(np.random.randint(0, 2, (self.input_size, self.neuron_count)), self.w1, other.w1)
        new_layer.w2 = np.where(np.random.randint(0, 2, (self.neuron_count, self.neuron_count)), self.w2, other.w2)
        new_layer.bias = np.where(np.random.randint(0, 2, self.neuron_count), self.bias, other.bias)
        new_layer.initial_cooldowns = np.where(np.random.randint(0, 2, self.neuron_count), self.initial_cooldowns, other.initial_cooldowns)
        new_layer.max_cooldowns = np.where(np.random.randint(0, 2, self.neuron_count), self.max_cooldowns, other.max_cooldowns)
        return new_layer
    
class BatchedLayer:
    def __init__(self, input_size, neuron_count, batch_size, activation_function):
        self.input_size = input_size
        self.neuron_count = neuron_count
        self.batch_size = batch_size
        self.activation_function = activation_function

        self.w1 = np.random.uniform(-1, 1, (input_size, neuron_count, batch_size))
        self.bias = np.random.uniform(-1, 1, (neuron_count, batch_size))

    def copy(self):
        new_layer = BatchedLayer(self.input_size, self.neuron_count, self.batch_size, self.activation_function)
        new_layer.w1 = self.w1.copy()
        new_layer.bias = self.bias.copy()
        return new_layer
    
    def initialize(self, batch_size=None):
        pass

    def batch_forward(self, value):
        """
        Perform a forward pass on a batch of input values, optimized for parallel network processing.

        Args:
            value (numpy.ndarray): The input values, with shape (batch_size, input_size).

        Returns:
            numpy.ndarray: The output values after applying the layer's activation function, with shape (batch_size, output_size, num_networks).
        """
        if len(value.shape) == 1:
            value = value.reshape(1, -1)
        if value.shape[1] != self.input_size:
            raise ValueError("Input size does not match layer input size")

        # Assuming self.w1 shape is (input_size, neuron_count, num_networks) and self.bias shape is (neuron_count, num_networks)
        # Adjust dimensions of value for broadcasting: (batch_size, input_size, 1)
        value_expanded = value[:, :, np.newaxis]

        # Dot product and add bias, utilizing broadcasting for parallel networks
        output = np.dot(value_expanded, self.w1) + self.bias

        # Apply activation function across all networks in parallel
        return self.activation_function.batch_forward(output)
    
    def mutate(self, weight_mutation_rate=0.1, weight_mutation_strength=0.1, bias_mutation_rate=0.1, bias_mutation_strength=0.1):
        if np.random.uniform(0, 1) < weight_mutation_rate:
            self.w1 += np.random.uniform(-weight_mutation_strength, weight_mutation_strength, (self.input_size, self.neuron_count, self.batch_size))
        if np.random.uniform(0, 1) < bias_mutation_rate:
            self.bias += np.random.uniform(-bias_mutation_strength, bias_mutation_strength, (self.neuron_count, self.batch_size))

    @staticmethod
    def batch_layers(*layers: 'Layer'):
        """
        Converts a list of layers to a batched layer.

        Args:
            layers (Layer): The layers to be batched.

        Returns:
            BatchedLayer: A batched layer that functions as multiple parallel networks.
        """
        # the networks must have the same input size, neuron count, and activation function, and must be of the same type
        if not all(layer.input_size == layers[0].input_size and layer.neuron_count == layers[0].neuron_count and layer.activation_function == layers[0].activation_function and type(layer) == type(layers[0]) for layer in layers):
            raise ValueError("All layers must have the same input size, neuron count, and type")
        # the batch size is the number of networks
        input_size = layers[0].input_size
        neuron_count = layers[0].neuron_count
        batch_size = len(layers)
        activation_function = layers[0].activation_function
        # make sure all layers are not subclasses of Layer and are only Layer. We already checked that they are of the same type, so checking the first one is enough
        if type(layers[0]) == Layer:
            batched_layer = BatchedLayer(input_size, neuron_count, batch_size, activation_function)
            batched_layer.w1 = np.stack([layer.w1 for layer in layers], axis=-1)
            batched_layer.bias = np.stack([layer.bias for layer in layers], axis=-1)
            return batched_layer
        elif type(layers[0]) == RGLayer:
            batched_layer = RGLayerBatched(input_size, neuron_count, batch_size, activation_function)
            batched_layer.w1 = np.stack([layer.w1 for layer in layers], axis=-1)
            batched_layer.w2 = np.stack([layer.w2 for layer in layers], axis=-1)
            batched_layer.bias = np.stack([layer.bias for layer in layers], axis=-1)
            batched_layer.stored_values = np.stack([layer.stored_values for layer in layers], axis=-1)
            batched_layer.propagating_values = np.stack([layer.propagating_values for layer in layers], axis=-1)
            batched_layer.max_cooldowns = np.stack([layer.max_cooldowns for layer in layers], axis=-1)
            batched_layer.initial_cooldowns = np.stack([layer.initial_cooldowns for layer in layers], axis=-1)
            batched_layer.current_cooldowns = np.stack([layer.current_cooldowns for layer in layers], axis=-1)
            return batched_layer
        else:
            raise ValueError("That type of layer is not supported for batching yet")
        
    def unbatch(self):
        """
        Converts a batched layer to a list of layers.

        Returns:
            list: A list of layers that were batched.
        """
        layers = [Layer(self.input_size, self.neuron_count, self.activation_function) for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            layers[i].w1 = self.w1[:, :, i]
            layers[i].bias = self.bias[:, i]
        return layers

        

class RGLayerBatched(BatchedLayer):
    def __init__(self, input_size, neuron_count, batch_size, activation_function):
        super().__init__(input_size, neuron_count, batch_size, activation_function)

        self.w2 = np.random.uniform(-1, 1, (neuron_count, neuron_count, batch_size))
        self.stored_values = np.zeros((neuron_count, batch_size))
        self.propagating_values = np.zeros((neuron_count, batch_size))
        self.max_cooldowns = np.random.randint(0, 10, (neuron_count, batch_size))
        self.initial_cooldowns = np.random.randint(0, 10, (neuron_count, batch_size))
        self.current_cooldowns = self.initial_cooldowns.copy()

    def copy(self):
        new_layer = RGLayerBatched(self.input_size, self.neuron_count, self.batch_size, self.activation_function)
        new_layer.w1 = self.w1.copy()
        new_layer.w2 = self.w2.copy()
        new_layer.bias = self.bias.copy()
        new_layer.stored_values = self.stored_values.copy()
        new_layer.propagating_values = self.propagating_values.copy()
        new_layer.max_cooldowns = self.max_cooldowns.copy()
        new_layer.initial_cooldowns = self.initial_cooldowns.copy()
        new_layer.current_cooldowns = self.current_cooldowns.copy()
        return new_layer

    def initialize(self, batch_size=None):
        if batch_size is None:
            self.stored_values = np.zeros((self.neuron_count, self.batch_size))
            self.propagating_values = np.zeros((self.neuron_count, self.batch_size))
        else:
            self.stored_values = np.zeros((batch_size, self.neuron_count, self.batch_size))
            self.propagating_values = np.zeros((batch_size, self.neuron_count, self.batch_size))

    def batch_forward(self, value):
        """
        Perform a forward pass for a batch of input values through the layer, optimized for parallel network processing with cooldowns.

        Args:
            value (numpy.ndarray): The input values for the layer.

        Returns:
            numpy.ndarray: The output values after applying the activation function.
        """
        if len(value.shape) == 1:
            value = value.reshape(1, -1)
        if value.shape[1] != self.input_size:
            raise ValueError("Input size does not match layer input size")

        # Vectorized operations for cooldown logic
        cooldown_zero_mask = self.current_cooldowns == 0
        self.propagating_values[:, cooldown_zero_mask] = self.stored_values[:, cooldown_zero_mask]

        # Prepare values for dot product by expanding for broadcasting if necessary
        value_expanded = value[:, :, np.newaxis]  # Adjust if your structure requires

        # Efficiently compute dot products and update stored values with vectorized operations
        self.stored_values[:, cooldown_zero_mask] = (np.dot(value_expanded, self.w1[:, cooldown_zero_mask])
                                                    + np.dot(self.propagating_values, self.w2[:, cooldown_zero_mask])
                                                    + self.bias[cooldown_zero_mask])

        # Decrement cooldowns and reset appropriately
        self.current_cooldowns -= 1
        self.current_cooldowns[self.current_cooldowns == -1] = self.max_cooldowns[self.current_cooldowns == -1]

        # Apply activation function across all networks in parallel
        return self.activation_function.batch_forward(self.propagating_values)

    def mutate(self, weight_mutation_rate=0.1, weight_mutation_strength=0.1, bias_mutation_rate=0.1, bias_mutation_strength=0.1, cooldown_mutation_rate=0.1, cooldown_mutation_strength=1):
        super().mutate(weight_mutation_rate, weight_mutation_strength, bias_mutation_rate, bias_mutation_strength)
        
        if np.random.uniform(0, 1) < weight_mutation_rate:
            self.w2 += np.random.uniform(-weight_mutation_strength, weight_mutation_strength, (self.neuron_count, self.neuron_count, self.batch_size))
        
        if np.random.uniform(0, 1) < cooldown_mutation_rate:
            self.initial_cooldowns += np.random.randint(-cooldown_mutation_strength, cooldown_mutation_strength, (self.neuron_count, self.batch_size))
        
        if np.random.uniform(0, 1) < cooldown_mutation_rate:
            self.max_cooldowns += np.random.randint(-cooldown_mutation_strength, cooldown_mutation_strength+1, (self.neuron_count, self.batch_size))

    def unbatch(self):
        layers = [RGLayer(self.input_size, self.neuron_count, self.activation_function) for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            layers[i].w1 = self.w1[:, :, i]
            layers[i].w2 = self.w2[:, :, i]
            layers[i].bias = self.bias[:, i]
            layers[i].stored_values = self.stored_values[:, i]
            layers[i].propagating_values = self.propagating_values[:, i]
            layers[i].max_cooldowns = self.max_cooldowns[:, i]
            layers[i].initial_cooldowns = self.initial_cooldowns[:, i]
            layers[i].current_cooldowns = self.current_cooldowns[:, i]
        return layers