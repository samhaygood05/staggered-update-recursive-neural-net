import numpy as np
import networkx as nx
from neuron import Neuron
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size, hidden_activation, output_activation, fire_cooldown=None, initial_cooldown=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # values: input, hidden, and output
        self.input_values = np.zeros(input_size)
        self.hiddens = [Neuron(hidden_activation, fire_cooldown, initial_cooldown) for _ in range(hidden_size)]
        self.outputs = [Neuron(output_activation, 0, 0) for _ in range(output_size)]

        # weights: input -> hidden, hidden -> hidden, and hidden -> output
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size)

    def add_hidden_node(self):
        self.hidden_size += 1
        #add it at a random position
        position = np.random.randint(0, len(self.hiddens))
        self.hiddens.insert(position, Neuron(self.hiddens[0].activation_function))
        self.W1 = np.insert(self.W1, position, np.random.randn(self.input_size), axis=1)
        self.W2 = np.insert(self.W2, position, np.random.randn(self.hidden_size), axis=0)
        self.W2 = np.insert(self.W2, position, np.random.randn(self.hidden_size+1), axis=1)

    def remove_hidden_node(self):
        if self.hidden_size > 1:
            #remove a random hidden node
            position = np.random.randint(0, len(self.hiddens))
            self.hiddens.pop(position)
            self.W1 = np.delete(self.W1, position, axis=1)
            self.W2 = np.delete(self.W2, position, axis=0)
            self.W2 = np.delete(self.W2, position, axis=1)
            self.hidden_size -= 1

    def reset(self):
        for hidden in self.hiddens:
            hidden.reset()
        for output in self.outputs:
            output.reset()

    def reset_batch(self, batch_size):
        for hidden in self.hiddens:
            hidden.reset_batch(batch_size)
        for output in self.outputs:
            output.reset_batch(batch_size)

    def copy(self):
        new_net = NeuralNet(self.input_size, self.hidden_size, self.output_size, self.hiddens[0].activation_function, self.outputs[0].activation_function)
        new_net.W1 = self.W1.copy()
        new_net.W2 = self.W2.copy()
        new_net.W3 = self.W3.copy()
        for i in range(self.hidden_size):
            new_net.hiddens[i].bias = self.hiddens[i].bias
            new_net.hiddens[i].fire_cooldown = self.hiddens[i].fire_cooldown
            new_net.hiddens[i].initial_cooldown = self.hiddens[i].initial_cooldown
        for i in range(self.output_size):
            new_net.outputs[i].bias = self.outputs[i].bias
            new_net.outputs[i].fire_cooldown = self.outputs[i].fire_cooldown
            new_net.outputs[i].initial_cooldown = self.outputs[i].initial_cooldown
        return new_net


    def visualize_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        #input nodes are green, hidden nodes are blue, output nodes are red
        #the weights are the edges, negative is red, positive is green, and the thickness is the magnitude
        for i in range(self.input_size):
            G.add_node(i, color='green')
        for i in range(self.hidden_size):
            G.add_node(i+self.input_size, color='blue')
        for i in range(self.output_size):
            G.add_node(i+self.input_size+self.hidden_size, color='red')
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                G.add_edge(i, j+self.input_size, weight=self.W1[i][j], color='green' if self.W1[i][j] > 0 else 'red', thickness=abs(self.W1[i][j]))
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                G.add_edge(i+self.input_size, j+self.input_size, weight=self.W2[i][j], color='green' if self.W2[i][j] > 0 else 'red', thickness=abs(self.W2[i][j]))
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                G.add_edge(i+self.input_size, j+self.input_size+self.hidden_size, weight=self.W3[i][j], color='green' if self.W3[i][j] > 0 else 'red', thickness=abs(self.W3[i][j]))

        # Define positions for input and output nodes
        fixed_positions = {}
        fixed_nodes = []
        y_input = 0
        y_output = 0
        input_step = 50
        output_step = 50

        for i in range(self.input_size):
            fixed_positions[i] = (-50, y_input)
            y_input += input_step
            fixed_nodes.append(i)

        for i in range(self.output_size):
            node_id = i + self.input_size + self.hidden_size
            fixed_positions[node_id] = (50, y_output)
            y_output += output_step
            fixed_nodes.append(node_id)

        # Apply spring layout for nodes that are not fixed (mostly hidden nodes)
        pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_nodes)
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        edge_colors = [G.edges[edge]['color'] for edge in G.edges()]
        edge_weights = [G.edges[edge]['thickness'] for edge in G.edges()]

        nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, width=edge_weights, with_labels=True, arrows=True)
        plt.show()

        return G

    def visualize_weights(self):
        # Visualize the weights
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs[0][0].imshow(self.W1, cmap='coolwarm', interpolation='nearest')
        axs[0][0].set_title('Input → Hidden Weights')
        axs[0][1].imshow(self.W2, cmap='coolwarm', interpolation='nearest')
        axs[0][1].set_title('Hidden → Hidden Weights')
        axs[0][2].imshow(self.W3, cmap='coolwarm', interpolation='nearest')
        axs[0][2].set_title('Hidden → Output Weights')

        # Visualize the fire cooldowns (fire_cooldown) and the next fire in (next_fire_in) as a 2xn matrix where the first row is the fire_cooldown and the second row is the next_fire_in in a single plot
        axs[1][0].imshow(np.array([[self.hiddens[i].fire_cooldown for i in range(self.hidden_size)], [self.hiddens[i].initial_cooldown for i in range(self.hidden_size)]]), cmap='coolwarm', interpolation='nearest')
        axs[1][0].set_title('Fire Cooldowns and Initial Cooldowns')


        # Visualize the biases
        axs[1][1].imshow(np.array([[self.hiddens[i].bias for i in range(self.hidden_size)]]), cmap='coolwarm', interpolation='nearest')
        axs[1][1].set_title('Hidden Biases')
        axs[1][2].imshow(np.array([[self.outputs[i].bias for i in range(self.output_size)]]), cmap='coolwarm', interpolation='nearest')
        axs[1][2].set_title('Output Biases')

        # Set ticks in increments of 1
        axs[0][0].set_xticks(np.arange(0, self.hidden_size, 1))
        axs[0][0].set_yticks(np.arange(0, self.input_size, 1))
        axs[0][1].set_xticks(np.arange(0, self.hidden_size, 1))
        axs[0][1].set_yticks(np.arange(0, self.hidden_size, 1))
        axs[0][2].set_xticks(np.arange(0, self.output_size, 1))
        axs[0][2].set_yticks(np.arange(0, self.hidden_size, 1))
        axs[1][0].set_xticks(np.arange(0, self.hidden_size, 1))
        axs[1][0].set_yticks(np.arange(0, 2, 1))
        axs[1][1].set_xticks(np.arange(0, self.hidden_size, 1))
        axs[1][1].set_yticks(np.arange(0, 1, 1))
        axs[1][2].set_xticks(np.arange(0, self.output_size, 1))
        axs[1][2].set_yticks(np.arange(0, 1, 1))

        plt.show()

    def visualize_values(self, use_raw_values=False, softmax_output=False):
        input_values = self.input_values
        if use_raw_values:
            hidden_values = [neuron.value for neuron in self.hiddens]
            output_values = [neuron.value for neuron in self.outputs]
        else:
            hidden_values = [neuron.forward() for neuron in self.hiddens]
            output_values = [neuron.forward() for neuron in self.outputs]

        if softmax_output:
            output_values = self.softmax(output_values)

        # Visualize the values
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].bar(np.arange(0, self.input_size, 1), self.input_values)
        axs[0].set_title('Input')
        axs[1].bar(np.arange(0, self.hidden_size, 1), hidden_values)
        axs[1].set_title('Hidden')
        axs[2].bar(np.arange(0, self.output_size, 1), output_values)
        axs[2].set_title('Output')

        # Set ticks in increments of 1
        axs[0].set_xticks(np.arange(0, self.input_size, 1))
        axs[1].set_xticks(np.arange(0, self.hidden_size, 1))
        axs[2].set_xticks(np.arange(0, self.output_size, 1))

        plt.show()

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self, input_values=None, softmax_output=False):
        if input_values is not None:
            self.input_values = np.array(input_values)
        else:
            self.input_values = np.zeros(self.input_size)
        hidden_values = [hidden.forward() for hidden in self.hiddens]

        # Compute the hidden layer
        hidden_from_input = np.dot(self.input_values, self.W1)
        hidden_from_hidden = np.dot(hidden_values, self.W2)
        hidden_values = hidden_from_input + hidden_from_hidden
        # update the hidden layer values
        for i, hidden in enumerate(self.hiddens):
            hidden.fire(hidden_values[i])

        # Compute the output layer
        output_values = np.dot(hidden_values, self.W3)

        # update the output layer values
        for i, output in enumerate(self.outputs):
            output.fire(output_values[i])

        # get the forward values
        outputs = [output.forward() for output in self.outputs]
        if softmax_output:
            outputs = self.softmax(outputs)
        return outputs
    
    def batch_forward(self, input_values=None, softmax_output=False):
        if input_values is not None:
            # Ensure input_values is a 2D array (batch_size, input_size)
            self.input_values = np.array(input_values)
            if self.input_values.ndim == 1:
                self.input_values = self.input_values[np.newaxis, :]
        else:
            self.input_values = np.zeros((1, self.input_size))
        
        batch_size = self.input_values.shape[0]
        
        # Initialize hidden_values for the entire batch
        hidden_values_batch = np.zeros((batch_size, len(self.hiddens)))
        
        for i, hidden in enumerate(self.hiddens):
            # Compute hidden.forward() for the entire batch
            hidden_values_batch[:, i] = hidden.forward()

        # Compute the hidden layer for the entire batch
        hidden_from_input = np.dot(self.input_values, self.W1)
        hidden_from_hidden = np.dot(hidden_values_batch, self.W2)
        hidden_values_batch = hidden_from_input + hidden_from_hidden
        
        # Update the hidden layer values for the entire batch
        for i, hidden in enumerate(self.hiddens):
            hidden.batch_fire(hidden_values_batch[:, i])
        
        # Compute the output layer for the entire batch
        output_values_batch = np.dot(hidden_values_batch, self.W3)
        
        # Update the output layer values for the entire batch
        for i, output in enumerate(self.outputs):
            output.batch_fire(output_values_batch[:, i])
        
        # Get the forward values for the entire batch
        outputs_batch = np.array([output.forward() for output in self.outputs]).T
        if softmax_output:
            outputs_batch = np.array([self.softmax(output) for output in outputs_batch])
        
        return outputs_batch

    def forward_n_times(self, n, input_values=None, softmax_output=False):
        # the input values should be a list of n input values if none, then convert it to a list of n None values
        output_values = []
        if input_values is None:
            input_values = [None] * n
        for i in range(n):
            output_values.append(self.forward(input_values[i], softmax_output))
        return output_values
    
    def batch_forward_n_times(self, n, input_values=None, softmax_output=False):
        # the input values should be a list of n input values if none, then convert it to a list of n None values
        output_values = []
        if input_values is None:
            input_values = [None] * n
        for i in range(n):
            output_values.append(self.batch_forward(input_values[i], softmax_output))
        return np.array(output_values)
    

    def mutate(self, weight_mutation_rate=0.1, weight_mutation_strength=0.1, bias_mutation_rate=0.1, bias_mutation_strength=0.1, fire_cooldown_mutation_rate=0.1, initial_cooldown_mutation_rate=0.1, hidden_node_count_mutation_rate=0.00):
        #weights and biases must be floats
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                if np.random.uniform(0, 1) < weight_mutation_rate:
                    self.W1[i][j] += np.random.uniform(-weight_mutation_strength, weight_mutation_strength)
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                if np.random.uniform(0, 1) < weight_mutation_rate:
                    self.W2[i][j] += np.random.uniform(-weight_mutation_strength, weight_mutation_strength)
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                if np.random.uniform(0, 1) < weight_mutation_rate:
                    self.W3[i][j] += np.random.uniform(-weight_mutation_strength, weight_mutation_strength)
        for i in range(self.hidden_size):
            self.hiddens[i].mutate(bias_mutation_rate, bias_mutation_strength, fire_cooldown_mutation_rate, initial_cooldown_mutation_rate)
        for i in range(self.output_size):
            self.outputs[i].mutate(bias_mutation_rate, bias_mutation_strength, fire_cooldown_mutation_rate, initial_cooldown_mutation_rate)

        if np.random.uniform(0, 1) < hidden_node_count_mutation_rate:
            if np.random.uniform(0, 1) < 0.5:
                self.add_hidden_node()
            else:
                self.remove_hidden_node()