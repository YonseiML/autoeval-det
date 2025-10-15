# import functools
# import warnings
# from collections import abc
# from inspect import getfullargspec
# from typing import Callable, Iterable, List, Optional

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn.parameter import Parameter

# from mmcv.utils import IS_NPU_AVAILABLE
# from mmengine.utils.dl_utils import TORCH_VERSION
# from mmengine.utils import digit_version
# from mmengine.utils.dist_utils import allreduce_grads as _allreduce_grads


import functools
import warnings
from collections import abc
from inspect import getfullargspec
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from mmcv.utils import IS_NPU_AVAILABLE, TORCH_VERSION, digit_version
from .utils.dist_utils import allreduce_grads as _allreduce_grads


try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.autocast would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    # Note that when PyTorch >= 1.6.0, we still cast tensor types to fp16
    # manually, so the behavior may not be consistent with real amp.
    if IS_NPU_AVAILABLE:
        from torch.npu.amp import autocast
    else:
        from torch.cuda.amp import autocast
except ImportError:
    pass


def wrap_fp16_model(model: nn.Module) -> None:
    """Wrap the FP32 model to FP16.

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.

    For PyTorch >= 1.6, this function will
    1. Set fp16 flag inside the model to True.

    Otherwise:
    1. Convert FP32 model to FP16.
    2. Remain some necessary layers to be FP32, e.g., normalization layers.
    3. Set `fp16_enabled` flag inside the model to True.

    Args:
        model (nn.Module): Model in FP32.
    """
    if (TORCH_VERSION == 'parrots'
            or digit_version(TORCH_VERSION) < digit_version('1.6.0')):
        # convert model to fp16
        model.half()
        # patch the normalization layers to make it work in fp32 mode
        patch_norm_fp32(model)
    # set `fp16_enabled` flag
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True