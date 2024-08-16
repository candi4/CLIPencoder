from typing import Union
import torch as th
import numpy as np
from PIL import Image
import os

def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device

def numpy_to_pil(array:np.ndarray):
    assert len(array.shape) in (2,3), f'array.shape={array.shape}'
    image = Image.fromarray(array)
    return image

### Directory

def upper_directory(filename, step=0) -> str:
    """ex. when step=0,
    directory/filename.exe -> directory
    dir1/dir2/ -> dir1/dir2
    dir1/dir2 -> dir1
    <SofaGuidewireNav>/SofaGW/utils.py -> <SofaGuidewireNav>/SofaGW
    """
    directory = os.path.dirname(filename)
    for i in range(step):
        directory = os.path.dirname(directory)
    return directory

# <VLMRM>
root_dir = upper_directory(os.path.abspath(__file__), 0)