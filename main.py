
if __name__ == "__main__":

    from training import *
    from neural_net import NeuralNet
    from activation import Activation
    import matplotlib.pyplot as plt


    simutation = Training()

    best_scores, worst_scores, best_network = simutation.train_n_times_batch(5000, 250)