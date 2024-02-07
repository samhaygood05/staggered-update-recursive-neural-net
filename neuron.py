import numpy as np

class Neuron:
    def __init__(self, activation_function, fire_cooldown=None, initial_cooldown=None):
        self.value = 0
        self.next_value = 0
        self.activation_function = activation_function
        self.fire_cooldown = 1
        self.initial_cooldown = 0
        self.current_cooldown = 0
        self.bias = 0
        self.initialize(fire_cooldown, initial_cooldown)

    def initialize(self, fire_cooldown, initial_cooldown):
        self.value = 0
        self.next_value = 0
        if fire_cooldown is not None:
            self.fire_cooldown = fire_cooldown
        else:
            self.fire_cooldown = np.random.randint(1, 10)
        if initial_cooldown is not None:
            self.initial_cooldown = initial_cooldown
        else:
            self.initial_cooldown = np.random.randint(0, self.fire_cooldown)

        self.current_cooldown = self.initial_cooldown
        self.bias = np.random.uniform(-1, 1)

    def reset(self):
        self.value = 0
        self.next_value = 0
        self.current_cooldown = self.initial_cooldown

    def reset_batch(self, batch_size):
        self.value = np.zeros(batch_size)
        self.next_value = np.zeros(batch_size)
        self.current_cooldown = np.ones(batch_size) * self.initial_cooldown

    def fire(self, value):
        if self.current_cooldown <= 0:
            self.value = self.next_value
            self.next_value = value + self.bias
            self.current_cooldown = self.fire_cooldown
        else:
            self.current_cooldown -= 1

    def batch_fire(self, value):
        self.value[self.current_cooldown <= 0] = self.next_value[self.current_cooldown <= 0]
        self.next_value[self.current_cooldown <= 0] = value[self.current_cooldown <= 0] + self.bias
        self.current_cooldown[self.current_cooldown > 0] -= 1

    def forward(self):
        return self.activation_function.forward(self.value)
    
    def mutate(self, bias_mutation_rate=0.1, bias_mutation_strength=0.1, fire_cooldown_mutation_rate=0.1, initial_cooldown_mutation_rate=0.1):
        #cooldowns must be positive integers
        if np.random.uniform(0, 1) < bias_mutation_rate:
            self.bias += np.random.uniform(-bias_mutation_strength, bias_mutation_strength)
        if np.random.uniform(0, 1) < fire_cooldown_mutation_rate:
            self.fire_cooldown = max(0, self.fire_cooldown + np.random.randint(-1, 2))
        if np.random.uniform(0, 1) < initial_cooldown_mutation_rate:
            self.initial_cooldown = max(0, self.initial_cooldown + np.random.randint(-1, 2))