# -*- utf-8 -*-
r"""
    custom_types.py
    ~~~~~~~~~~~~~~~

    A set of compact re-usable type definitions.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: , see MIT for more details.
"""

# pylint: disable=duplicate-code
from argparse import Namespace
from pathlib import Path
import sys
from typing import Any, Callable, List, Optional, Set, Tuple, Union

try:  # noinspection PyUnresolvedReferences
    import numpy as np
except ImportError:
    pass

try:  # noinspection PyUnresolvedReferences
    from pandas import DataFrame
except ImportError:
    pass

try:  # noinspection PyUnresolvedReferences
    from fastai.basic_data import DataBunch
except ImportError:
    pass
try:  # noinspection PyUnresolvedReferences
    from torch import Tensor
except ImportError:
    pass


def _has_module(mod_name: str) -> bool:
    r""" Returns \p True if \p mod_name has been imported """
    return mod_name in sys.modules


# Stores whether each module is available
has_np = _has_module("numpy")
has_pd = _has_module("pandas")
has_fastai = _has_module("fastai")
has_torch = _has_module("torch")

OptBool = Optional[bool]
OptCallable = Optional[Callable]
if has_fastai:
    OptDataBunch = Optional[DataBunch]
if has_pd:
    OptDataFrame = Optional[DataFrame]
OptDict = Optional[dict]
OptFloat = Optional[float]
OptInt = Optional[int]
OptListInt = Optional[List[int]]
OptListStr = Optional[List[str]]
OptNamespace = Optional[Namespace]
OptStr = Optional[str]
if has_torch:
    OptTensor = Optional[Tensor]

ListOrInt = Union[int, List[int]]
SetListOrInt = Union[int, Set[int], List[int]]
SetOrList = Union[List[Any], Set[Any]]

PathOrStr = Union[Path, str]

if has_torch:
    TensorTuple = Tuple[Tensor, Tensor]
    if has_np:
        TorchOrNp = Union[Tensor, np.ndarray]
