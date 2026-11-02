"""Device selection for APDTFlow.

One place that turns a user-facing ``device`` argument into a
:class:`torch.device`, so the forecaster, the CLI and the benchmarks all agree
on what ``"auto"`` means.
"""

from typing import Optional, Union

import torch

DeviceLike = Union[str, torch.device, None]

#: Accepted string values besides explicit ones such as ``"cuda:1"``.
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def mps_is_available() -> bool:
    """True when PyTorch was built with MPS support and an MPS device exists."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(device: DeviceLike = "auto") -> torch.device:
    """Resolve a device argument to a concrete :class:`torch.device`.

    ``"auto"`` (or ``None``) picks the first available of CUDA, Apple MPS and
    CPU. Explicit requests are validated: asking for ``"cuda"`` or ``"mps"``
    on a machine without that backend raises a ``ValueError`` that says how
    to fix it, rather than failing later inside ``fit``.

    Parameters
    ----------
    device : {"auto", "cuda", "mps", "cpu"}, str, torch.device or None
        ``"cuda:1"``-style indices are accepted.

    Returns
    -------
    torch.device
    """
    if device is None:
        device = "auto"
    if isinstance(device, torch.device):
        requested = device
    else:
        name = str(device).strip().lower()
        if name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if mps_is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            requested = torch.device(name)
        except RuntimeError as exc:
            raise ValueError(
                f"Unknown device {device!r}; expected one of {DEVICE_CHOICES} "
                f"or an indexed device such as 'cuda:1'."
            ) from exc

    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "device='cuda' was requested but torch.cuda.is_available() is "
                "False. Install a CUDA build of PyTorch "
                "(https://pytorch.org/get-started/locally/) or use "
                "device='auto' / 'cpu'."
            )
        if requested.index is not None and requested.index >= torch.cuda.device_count():
            raise ValueError(
                f"device={str(requested)!r} was requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
    elif requested.type == "mps":
        if not mps_is_available():
            raise ValueError(
                "device='mps' was requested but the Apple MPS backend is not "
                "available (it needs Apple silicon and a PyTorch build with "
                "MPS). Use device='auto' or 'cpu'."
            )
    elif requested.type != "cpu":
        raise ValueError(
            f"Unsupported device type {requested.type!r}; APDTFlow supports "
            f"{DEVICE_CHOICES}."
        )
    return requested


def describe_device(device: Optional[torch.device] = None) -> str:
    """Human-readable name of a device, e.g. ``'cuda (NVIDIA GeForce ...)'``."""
    device = resolve_device(device)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda ({torch.cuda.get_device_name(index)})"
    return device.type
