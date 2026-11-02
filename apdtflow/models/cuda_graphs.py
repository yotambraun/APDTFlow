"""Training steps replayed as one captured CUDA graph.

APDTFlow's forward pass is a long chain of small operations: two RK4 solves,
each step evaluating a small MLP on a ``(batch, hidden_dim)`` tensor. At the
default size a training step launches about 20,000 GPU kernels that each run
for a few microseconds, so an eager GPU step is bound by kernel-launch
overhead, not arithmetic, and the GPU sits idle most of the time.

A CUDA graph records every kernel of one optimizer step (forward, backward
and Adam) once and replays them with a single launch. The arithmetic is
unchanged; only the per-kernel CPU overhead disappears. This follows the
whole-iteration recipe in the PyTorch CUDA-graphs guide: warm up on a side
stream, capture with static input buffers, then copy each batch into those
buffers and replay.

Requirements, all checked by :func:`graph_capture_supported` before use:
fixed input shapes (batches smaller than the captured one run eagerly on the
same parameters and optimizer state), no host synchronisation inside the
step, and an optimizer created with ``capturable=True``.
"""

from typing import Callable, NamedTuple, Optional

import torch

__all__ = ["CUDAGraphTrainStep", "graph_capture_supported"]


def graph_capture_supported(device: torch.device, ode_method: str, has_exog: bool) -> bool:
    """Whether a training step on this configuration can be captured.

    Adaptive solvers (``dopri5_adjoint``) choose step sizes on the host, so
    they cannot be captured. Exogenous-input fusion is not covered by the
    capture tests, so it keeps the eager path.
    """
    return device.type == "cuda" and ode_method == "rk4" and not has_exog


class _Captured(NamedTuple):
    graph: "torch.cuda.CUDAGraph"
    x: torch.Tensor  # static input buffers the graph reads
    y: torch.Tensor
    loss: torch.Tensor  # static output the graph writes


class CUDAGraphTrainStep:
    """Run optimizer steps as replays of a captured CUDA graph.

    Args:
        loss_fn: ``loss_fn(x, y) -> scalar loss`` for one batch. It must not
            read values back to the host.
        optimizer: The optimizer stepping the model's parameters, created with
            ``capturable=True``.
        batch_size: The full batch size; the graph is captured on the first
            batch of this size after warm-up.
        warmup_steps: Eager steps run on a side stream before capture, as the
            PyTorch guide requires. They are real training steps.

    Call the instance once per batch; it returns the batch loss as a new
    0-dim tensor on the device. Batches of another shape (the last, partial
    batch of an epoch) run eagerly on the same parameters and optimizer state.
    """

    def __init__(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        warmup_steps: int = 3,
    ):
        for group in optimizer.param_groups:
            if not group.get("capturable", False):
                raise ValueError("CUDAGraphTrainStep needs an optimizer created with capturable=True")
        self._batch_size = batch_size
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._warmup_left = warmup_steps
        self._captured: Optional[_Captured] = None
        self._side_stream = torch.cuda.Stream()

    @property
    def captured(self) -> bool:
        """Whether the step has been captured and is now replayed."""
        return self._captured is not None

    def _eager(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Zero in place rather than setting to None: after capture the
        # gradients live in the graph's memory pool, and the captured
        # optimizer step reads exactly those tensors.
        self._optimizer.zero_grad(set_to_none=False)
        loss = self._loss_fn(x, y)
        loss.backward()
        self._optimizer.step()
        return loss.detach()

    def _capture(self, x: torch.Tensor, y: torch.Tensor) -> _Captured:
        static_x, static_y = x.clone(), y.clone()
        # Gradients are allocated inside the graph's pool on capture; the
        # captured backward then overwrites them on every replay.
        self._optimizer.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = self._loss_fn(static_x, static_y)
            loss.backward()
            self._optimizer.step()
        return _Captured(graph, static_x, static_y, loss.detach())

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        captured = self._captured
        if captured is None:
            if self._warmup_left > 0 or x.shape[0] != self._batch_size:
                # Warm-up, and any smaller batch before capture, run eagerly
                # on the side stream; the graph is only captured at full size.
                self._warmup_left = max(0, self._warmup_left - 1)
                stream = self._side_stream
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    loss = self._eager(x, y)
                torch.cuda.current_stream().wait_stream(stream)
                return loss
            captured = self._captured = self._capture(x, y)
        elif x.shape != captured.x.shape or y.shape != captured.y.shape:
            return self._eager(x, y)
        captured.x.copy_(x, non_blocking=True)
        captured.y.copy_(y, non_blocking=True)
        captured.graph.replay()
        return captured.loss.clone()
