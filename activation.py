import numpy as np
from typing import Callable, Optional

class Activation:
    def __init__(self, forward: Callable, batch_forward: Optional[Callable] = None) -> None:
        self.forward = forward
        self.batch_forward = forward if batch_forward is None else batch_forward