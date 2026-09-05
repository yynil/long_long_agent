"""Match the official BF16 GEMM path for short inference matrix multiplies.

Only independent matrix rows are aligned. Recurrent inputs, timestep counts,
state and losses are never padded here. Training retains its original path.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_REFERENCE_ROWS = ContextVar("diagnostic_reference_rows", default=0)


@contextmanager
def diagnostic_reference_rows(rows):
    """Reproduce one GEMM shape for diagnosis, never enable as a deployment fix."""
    if type(rows) is not int or rows <= 0:
        raise ValueError("reference rows must be positive")
    token = _REFERENCE_ROWS.set(rows)
    try:
        yield
    finally:
        _REFERENCE_ROWS.reset(token)


def _aligned_rows(x):
    rows = x.numel() // x.shape[-1]
    reference_rows = _REFERENCE_ROWS.get()
    if (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and not torch.is_grad_enabled()
        and rows < reference_rows
    ):
        return torch.nn.functional.pad(
            x.reshape(rows, x.shape[-1]), (0, 0, 0, reference_rows - rows)
        ), rows
    return None, rows


def inference_matmul(x, weight):
    aligned, rows = _aligned_rows(x)
    if aligned is None:
        return x @ weight
    return (aligned @ weight)[:rows].reshape(*x.shape[:-1], weight.shape[-1])


def inference_linear(module, x):
    aligned, rows = _aligned_rows(x)
    if aligned is None:
        return module(x)
    result = module(aligned)
    return result[:rows].reshape(*x.shape[:-1], result.shape[-1])
