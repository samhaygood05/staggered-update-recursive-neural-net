import numpy as np

class Activation:
    def __init__(self, forward, batch_forward = None):
        self.forward = forward
        self.batch_forward = forward if batch_forward is None else batch_forward