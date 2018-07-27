import numpy as np
from collections import deque


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        if buffer_size < batch_size:
            batch_size = buffer_size

        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        samples = [self.buffer[i] for i in index]
        return zip(*samples)