import enum
import os
from dataclasses import dataclass
from numpy.typing import NDArray
from pathlib import Path
from ...absolute_path import absolute_path, DATA_PATH


def choose_device(self) -> str:
    """
    Automatically select which PyTorch device to use.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_point_cloud(self):
    pc_path = Path(absolute_path(DATA_PATH))/'pc'
    pc_list = sorted(pc_path.glob('*.npz'), key=os.path.getmtime, reverse=True)
    return [pc.name for pc in pc_list]


@dataclass
class MeshGenerationResult:
    verts: NDArray | None
    faces: NDArray | None
    step: int
    final: bool
    status: str = ''


class StepPreviewMode(enum.Enum):
    NONE = "None"
    FAST = "Fast"
    ACCURATE = "Accurate"