"""
Code for implementing LRP.
"""
import numpy as np


def lrp_linear_single(x: np.ndarray, y: np.ndarray, rel_y: np.ndarray,
                      w: np.ndarray = None, eps: float = 0.001) -> np.ndarray:
    """
    Implements the LRP-epsilon rule for a linear layer: y = w @ x + b.
    This function only computes LRP for a single example.
    
    :param x: Layer input (input_size,)
    :param y: Layer output (output_size,)
    :param rel_y: Network output relevance (output_size,)
    :param w: Weight matrix (input_size, output_size). If left blank,
        the weight matrix is assumed to be the identity: y = x + b
    :param eps: Stabilizer
    :return: The relevance of x (input_size,)
    """
    denom = y + eps * np.where(y >= 0, 1., -1.)
    if w is None:
        return x * (rel_y / denom)
    return (w * x[:, np.newaxis]) @ (rel_y / denom)


def lrp_linear(x: np.ndarray, y: np.ndarray, rel_y: np.ndarray,
               w: np.ndarray = None, eps: float = 0.001) -> np.ndarray:
    """
    Implements the LRP-epsilon rule for a linear layer: y = w @ x + b.

    :param x: Input (batch_size, input_size)
    :param y: Output (batch_size, output_size)
    :param rel_y: Network output relevance (batch_size, output_size)
    :param w: Transposed weight matrix (input_size, output_size). If
        left blank, the weight matrix is assumed to be the identity:
        y = x + b
    :param eps: Stabilizer
    :return: The relevance of x (batch_size, input_size)
    """
    y = y + eps * np.where(y >= 0, 1., -1.)
    if w is None:
        return x * (rel_y / y)

    lhs = w[np.newaxis, :, :] * x[:, :, np.newaxis]
    rhs = (rel_y / y)[:, :, np.newaxis]
    return (lhs @ rhs).squeeze(2)


def lrp_matmul(x: np.ndarray, w: np.ndarray, y: np.ndarray, rel_y: np.ndarray,
               eps: float = 0.001) -> np.ndarray:
    """
    LRP-epsilon for a matrix multiplication layer: y = w @ x. One of the
    two matrices is treated as a weight matrix. All matrices are assumed
    to be batched.

    :param x: Input (..., m, n)
    :param w: Weight matrix (..., p, m)
    :param y: Output (..., p, n)
    :param rel_y: Output relevance (..., p, n)
    :param eps: Stabilizer
    :return: The relevance of x (..., m, n)
    """
    w = w.swapaxes(-1, -2)
    y = y + eps * np.where(y >= 0, 1., -1.)

    lhs = np.moveaxis(w[..., np.newaxis] * x[..., np.newaxis, :], -1, -3)
    rhs = (rel_y / y).swapaxes(-1, -2)[..., np.newaxis]
    return (lhs @ rhs).squeeze(-1).swapaxes(-1, -2)
