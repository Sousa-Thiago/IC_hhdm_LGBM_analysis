import numpy as np


def batch_generator(data, batch_size):
    """
    Generate batches of data.
    Given a list of numpy data, it iterates over the list and returns batches of the same size

    Args:
        data (np.array): Data array
        batch_size (int): Size of each batch

    Yields:
        np.array: Batch
    """
    indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr
