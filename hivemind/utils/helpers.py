import torch
import os
from pathlib import Path

def get_device():
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {x}"
        size /= 1024.0
    return f"{size:.2f} PB"

def ensure_directory(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)
