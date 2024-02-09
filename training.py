try:
    # Try to import CuPy
    import cupy as cp
    # Attempt to allocate memory on a GPU to confirm its presence
    cp.array([1])
    # If successful, alias CuPy as np to use it as if it were NumPy
    np = cp
    print("Using CuPy")
except (ImportError):
    # If CuPy is not installed, fall back to NumPy
    import numpy as np
    print("CuPy not found, using NumPy")
except (cp.cuda.runtime.CUDARuntimeError):
    # If no GPU is found, fall back to NumPy
    import numpy as np
    print("No GPU found, using NumPy")

from neural_net import NeuralNet
from layers import Layer, RGLayer
from concurrent.futures import ProcessPoolExecutor
from activation import Activation
import time
from datetime import timedelta

def convert_to_sequence(outputs):
    # get the maximum value of each row in the output
    max_indices = np.argmax(outputs, axis=1)

    # cluster the outputs by stating how many times a value is repeated in a row to condense it ex: [4, 4, 4, 4, 2, 2, 2, 4, 1, 1] -> [(4,4), (3,2), (1, 4), (2,1)]
    clustered = []
    last_value = -1
    for i in range(len(max_indices)):
        if last_value == -1:
            last_value = max_indices[i]
            count = 1
        elif max_indices[i] == last_value:
            count += 1
        else:
            clustered.append((count, last_value))
            last_value = max_indices[i]
            count = 1
        if i == len(max_indices) - 1:
            clustered.append((count, last_value))

    return clustered

def generate_random_sequence(num_classes, number, length=None):
    #must not have a number repeat twice in a row
    sequence = []
    for i in range(number):
        next_value = np.random.randint(0, num_classes)
        while len(sequence) > 0 and sequence[-1] == next_value:
            next_value = np.random.randint(0, num_classes)
        if length is None:
            length = np.random.randint(1, 10)
        sequence.append((length, next_value))
    return sequence

def convert_sequence_to_input(sequence, num_classes):
    input1 = []

    for i in range(len(sequence)):
        input1.extend([sequence[i][1]] * sequence[i][0])

    # convert to one hot encoding (-1 should be treated as full 0)
    one_hot = np.zeros((len(input1), num_classes))
    for i in range(len(input1)):
        if input1[i] != -1:
            one_hot[i][input1[i]] = 1
    return one_hot

def score_sequence(sequence1, sequence2, correct_length_score=0.25, correct_value_score=1):
    #the score should be +1 for each correct value and it should be scored based on the distance between the correct length and the actual length
    best_possible_score = len(sequence2) * (correct_value_score + correct_length_score)
    if len(sequence1) != len(sequence2):
        return 0
    score = 0
    for i in range(len(sequence1)):
        if sequence1[i][1] == sequence2[i][1]:
            score += correct_value_score
            length_difference = abs(sequence1[i][0] - sequence2[i][0])
            score += correct_length_score * (1 - length_difference / sequence2[i][0])

    # normalize the score to be between 0 and 1
    try:
        return score / best_possible_score
    except ZeroDivisionError:
        return -1

def score_full_sequence(sequence1, sequence2, correct_length_score=0.25, correct_value_score=1):
    if len(sequence1) == 0:
        return 0
    # if a sequences is longer than the other, find the adjacent subsequence with the highest score
    if len(sequence1) > len(sequence2):
        max_score = 0
        for i in range(len(sequence1) - len(sequence2) + 1):
            score = score_sequence(sequence1[i:i + len(sequence2)], sequence2, correct_length_score, correct_value_score)
            if score > max_score:
                max_score = score
        return max_score
    elif len(sequence2) > len(sequence1):
        best_possible_score = len(sequence2) * (correct_value_score + correct_length_score)
        subsequence_score = len(sequence1) * (correct_value_score + correct_length_score)
        max_score = 0
        for i in range(len(sequence2) - len(sequence1) + 1):
            subsequence = sequence2[i:i + len(sequence1)]
            score = score_sequence(sequence1, subsequence, correct_length_score, correct_value_score)

            # normalize the score for the actual length
            score = score * subsequence_score / best_possible_score
            if score > max_score:
                max_score = score
        return max_score
    else:
        return score_sequence(sequence1, sequence2)
    
def score_output(output, sequence, correct_length_score=0.25, correct_value_score=1):
    sequence_output = convert_to_sequence(output)
    # remove any part that has a duratino of less than or equal to 3
    sequence_output = [x for x in sequence_output if x[0] > 3]
    return score_full_sequence(sequence_output, sequence, correct_length_score, correct_value_score)

def sigmoid_forward(x):
    return 1/(1+np.exp(-x))

def softmax_forward(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

class Training:
    def __init__(self, network_count=100):
        self.networks = []
        self.network_count = network_count
        sigmoid = Activation(forward=sigmoid_forward)
        softmax = Activation(forward=softmax_forward)
        for i in range(network_count):
            self.networks.append(NeuralNet(5, (50, RGLayer, sigmoid), (5, Layer, softmax)))

    def train_network(self, network_index, inputs, original_sequences, ticks, batch_size):
        network = self.networks[network_index]
        network.initialize(batch_size)
        output = network.batch_forward_n_times(inputs, ticks)
        # remove first 25 outputs
        output = output[25:]
        total_score = sum(score_output(output[j], original_sequences[j]) for j in range(batch_size))
        average_score = total_score / batch_size
        return network_index, average_score

    def train(self, ticks, shots=10, elimination_rate=0.5, generation=None, total_generations=None):
        sequences = [generate_random_sequence(5, 5, 5) for i in range(shots)]
        # make a copy of the sequence
        original_sequences = [x.copy() for x in sequences]
        for x in sequences:
            x.append((ticks-25, -1))
        inputs = [convert_sequence_to_input(x, 5) for x in sequences]
        #initialize scores with 0s
        scores = np.zeros(self.network_count)
        for i, network in enumerate(self.networks):
            if generation is not None and total_generations is not None:
                print(f"Training network {i+1}/{self.network_count} (generation {generation}/{total_generations})", end="\r")
            else:
                print(f"Training network {i+1}/{self.network_count}")
            for j in range(shots):
                output = network.forward_n_times(inputs[j], ticks)
                # remove first 25 outputs
                output = output[25:]
                total_score = 0
                total_score += score_output(output, original_sequences[j])
            average_score = total_score / shots
            scores[i] += average_score
        sorted_indices = np.argsort(scores)
        best_score = scores[sorted_indices[-1]]
        best_network = self.networks[sorted_indices[-1]]
        worst_score = scores[sorted_indices[0]]
        # remove the worst networks and replace them with the best ones
        for i in range(int(elimination_rate * self.network_count)):
            self.networks[sorted_indices[i]] = self.networks[sorted_indices[-i-1]].copy()

        # mutate the networks
        for network in self.networks:
            network.mutate()

        return best_score, worst_score, best_network
    
    def train_batch(self, ticks, batch_size=10, elimination_rate=0.5, generation=None, total_generations=None):
        sequences = [generate_random_sequence(5, 5, 5) for i in range(batch_size)]
        # make a copy of the sequence
        original_sequences = [x.copy() for x in sequences]
        for x in sequences:
            x.append((ticks-25, -1))
        inputs = [convert_sequence_to_input(x, 5) for x in sequences]
        inputs = np.array(inputs)
        # currently inputs dimension is (batch_size, sequence_length, num_classes)
        # we need to change it to (sequence_length, batch_size, num_classes)
        inputs = np.swapaxes(inputs, 0, 1)

        #initialize scores with 0s
        scores = np.zeros(self.network_count)
        for i, network in enumerate(self.networks):
            network.reset_batch(batch_size)
            output = network.batch_forward_n_times(inputs, ticks)
            # remove first 25 outputs
            output = output[25:]
            total_score = 0
            for j in range(batch_size):
                total_score += score_output(output[j], original_sequences[j])
            average_score = total_score / batch_size
            scores[i] += average_score
        sorted_indices = np.argsort(scores)
        best_score = scores[sorted_indices[-1]]
        best_network = self.networks[sorted_indices[-1]]
        worst_score = scores[sorted_indices[0]]
        # remove the worst networks and replace them with the best ones
        for i in range(int(elimination_rate * self.network_count)):
            self.networks[sorted_indices[i]] = self.networks[sorted_indices[-i-1]].copy()

        # mutate the networks
        for network in self.networks:
            network.mutate()

        return best_score, worst_score, best_network
    
    def train_batch_parallel(self, ticks, batch_size=10, elimination_rate=0.5, generation=None, total_generations=None):
        sequences = [generate_random_sequence(5, 5, 5) for i in range(batch_size)]
        # make a copy of the sequence
        original_sequences = [x.copy() for x in sequences]
        for x in sequences:
            x.append((ticks-25, -1))
        inputs = [convert_sequence_to_input(x, 5) for x in sequences]
        inputs = np.array(inputs)
        # currently inputs dimension is (batch_size, sequence_length, num_classes)
        # we need to change it to (sequence_length, batch_size, num_classes)
        inputs = np.swapaxes(inputs, 0, 1)

        #initialize scores with 0s
        scores = np.zeros(self.network_count)

        # Prepare data for parallel execution
        futures = []
        with ProcessPoolExecutor() as executor:
            for i in range(self.network_count):
                future = executor.submit(self.train_network, i, inputs, original_sequences, ticks, batch_size)
                futures.append(future)

            for future in futures:
                network_index, average_score = future.result()
                scores[network_index] += average_score

        sorted_indices = np.argsort(scores)
        best_score = scores[sorted_indices[-1]]
        best_network = self.networks[sorted_indices[-1]]
        worst_score = scores[sorted_indices[0]]
        # remove the worst networks and replace them with the best ones
        for i in range(int(elimination_rate * self.network_count)):
            self.networks[sorted_indices[i]] = self.networks[sorted_indices[-i-1]].copy()

        # mutate the networks
        for network in self.networks:
            network.mutate()

        return best_score, worst_score, best_network
    
    def train_n_times(self, n, ticks, shots=10, elimination_rate=0.5):
        start_time = time.time()
        best_scores = []
        worst_scores = []
        for i in range(n):
            best_score, worst_score, best_network = self.train(ticks, shots, elimination_rate, i, n)
            best_scores.append(best_score)
            worst_scores.append(worst_score)

        dt = time.time() - start_time

        # format the time to be h:m:s
        dt = timedelta(seconds=dt)

        print(f"Training Completed in {dt}")
        print("""  _____   ____  _   _ ______ 
 |  __ \ / __ \| \ | |  ____|
 | |  | | |  | |  \| | |__   
 | |  | | |  | | . ` |  __|  
 | |__| | |__| | |\  | |____ 
 |_____/ \____/|_| \_|______|""")
        return best_scores, worst_scores, best_network
    
    def train_n_times_batch(self, n, ticks, batch_size=10, elimination_rate=0.5):
        start_time = time.time()
        best_scores = []
        worst_scores = []
        dts = []
        for i in range(n):
            gen_start_time = time.time()
            best_score, worst_score, best_network = self.train_batch(ticks, batch_size, elimination_rate, i, n)
            best_scores.append(best_score)
            worst_scores.append(worst_score)
            dt = round(time.time() - gen_start_time, 2)
            dts.append(dt)
            avg_dt = round(sum(dts)/len(dts), 2)
            estimate = round(avg_dt * (n - i-1), 0)
            print(f"Generation {i+1}/{n} Completed in {dt} ({avg_dt}) - Estimated Time Remaining: {timedelta(seconds=estimate)}     ", end="\r")

        dt = round(time.time() - start_time, 0)

        # format the time to be h:m:s
        dt = timedelta(seconds=dt)

        print(f"Training Completed in {dt} (~{avg_dt} per generation)")
        print("""  _____   ____  _   _ ______ 
 |  __ \ / __ \| \ | |  ____|
 | |  | | |  | |  \| | |__   
 | |  | | |  | | . ` |  __|  
 | |__| | |__| | |\  | |____ 
 |_____/ \____/|_| \_|______|""")
        return best_scores, worst_scores, best_network
    
    def train_n_times_batch_parallel(self, n, ticks, batch_size=10, elimination_rate=0.5):
        start_time = time.time()
        best_scores = []
        worst_scores = []
        dts = []
        for i in range(n):
            gen_start_time = time.time()
            best_score, worst_score, best_network = self.train_batch_parallel(ticks, batch_size, elimination_rate, i, n)
            best_scores.append(best_score)
            worst_scores.append(worst_score)
            dt = round(time.time() - gen_start_time, 2)
            dts.append(dt)
            avg_dt = round(sum(dts)/len(dts), 2)
            estimate = round(avg_dt * (n - i-1), 0)
            print(f"Generation {i+1}/{n} Completed in {dt} ({avg_dt}) - Estimated Time Remaining: {timedelta(seconds=estimate)}     ", end="\r")

        dt = round(time.time() - start_time, 0)

        # format the time to be h:m:s
        dt = timedelta(seconds=dt)

        print(f"Training Completed in {dt} (~{avg_dt} per generation)                        ")
        print("""  _____   ____  _   _ ______ 
 |  __ \ / __ \| \ | |  ____|
 | |  | | |  | |  \| | |__   
 | |  | | |  | | . ` |  __|  
 | |__| | |__| | |\  | |____ 
 |_____/ \____/|_| \_|______|""")
        return best_scores, worst_scores, best_network