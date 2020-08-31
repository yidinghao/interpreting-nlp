"""
Code for implementing LRP.
"""
import numpy as np


def lrp_linear_single(x: np.ndarray, y: np.ndarray, rel_y: np.ndarray,
                      w: np.ndarray = None, eps: float = 0.001) -> np.ndarray:
    """
    Implements the LRP-epsilon rule for a linear layer.
    
    :param x: Layer input (input_size,).
    :param w: Weight matrix (input_size, output_size).
    :param y: Layer output (output_size,).
    :param rel_y: Network output relevance (output_size,).
    :param eps: Stabilizer.
    :return: The relevance of x (input_size,).
    """
    denom = y + eps * np.where(y >= 0, 1., -1.)
    if w is None:
        return x * (rel_y / denom)
    else:
        return np.matmul(w * x[:, np.newaxis], rel_y / denom)


def lrp_linear(x: np.ndarray, y: np.ndarray, rel_y: np.ndarray,
               w: np.ndarray = None, eps: float = 0.001) -> np.ndarray:
    """
    Implements the LRP-epsilon rule for a linear layer.

    :param x: Layer input (batch_size, input_size).
    :param w: Weight matrix (input_size, output_size).
    :param y: Layer output (batch_size, output_size).
    :param rel_y: Network output relevance (batch_size, output_size).
    :param eps: Stabilizer.
    :return: The relevance of x (batch_size, input_size).
    """
    denom = y + eps * np.where(y >= 0, 1., -1.)
    if w is None:
        return x * (rel_y / denom)

    lhs = w[np.newaxis, :, :] * x[:, :, np.newaxis]
    rhs = (rel_y / denom)[:, :, np.newaxis]
    return (lhs @ rhs).squeeze(2)
