import torch

def save_checkpoint(model, optimizer, epoch, filename):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


def load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """Seed all RNGs for reproducible runs.

    With ``deterministic=True`` (the default) PyTorch is additionally
    switched to deterministic algorithms and a single CPU thread, trading
    speed for bit-for-bit reproducibility — the configuration used for the
    published benchmark numbers.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
