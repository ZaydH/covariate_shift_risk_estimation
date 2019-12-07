from argparse import Namespace
from enum import Enum
from pathlib import Path
import re
import socket
import time
from typing import Callable, Optional, Set, Tuple, Union

import torch
from fastai.basic_data import DeviceDataLoader
from torch import Tensor
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data import Dataset as TextDataset
from torchtext.data import Iterator


def _check_is_talapas() -> bool:
    r""" Returns \p True if running on talapas """
    host = socket.gethostname().lower()
    if "talapas" in host:
        return True
    if re.match(r"^n\d{3}$", host):
        return True

    num_string = r"(\d{3}|\d{3}-\d{3})"
    if re.match(f"n\\[{num_string}(,{num_string})*\\]", host):
        return True
    return False


BASE_DIR = Path(".").absolute() if not _check_is_talapas() else Path("/home/zhammoud/projects/nlp")
DATA_DIR = BASE_DIR / ".data"

IS_CUDA = torch.cuda.is_available()
# if IS_CUDA:
#     # noinspection PyUnresolvedReferences
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)
TORCH_DEVICE = torch.device("cuda:0" if IS_CUDA else "cpu")

NUM_WORKERS = 0 if IS_CUDA else 0  # For stability on a Mac/GPU

DatasetType = Union[TensorDataset, TextDataset]
LoaderType = Union[DeviceDataLoader, Iterator]

POS_LABEL = 1
U_LABEL = 0
NEG_LABEL = -1


def build_loss_functions(pos_classes: Optional[Union[Set[int], int]] = None) \
        -> Tuple[Callable, Callable]:
    r"""
    Constructor method for basic losses, specifically the logistic and sigmoid losses.

    :param pos_classes: Set of (mapped) class labels to treat as "positive"  If not specified,
                        then return the univariate version of the losses.
    :return: Logistic and sigmoid loss functions, respectively.
    """
    if pos_classes is None:
        def _logistic_loss_univariate(in_tensor: Tensor) -> Tensor:
            return -F.logsigmoid(in_tensor)

        def _sigmoid_loss_univariate(in_tensor: Tensor) -> Tensor:
            return torch.sigmoid(-in_tensor)

        return _logistic_loss_univariate, _sigmoid_loss_univariate

    if isinstance(pos_classes, int):
        pos_classes = {pos_classes}

    def _build_y_tensor(target: Tensor) -> Tensor:
        r""" Create a y vector from target since may change labels """
        y = torch.full(target.shape, -1).cuda(TORCH_DEVICE)
        for pos_lbl in pos_classes:
            y[target == pos_lbl] = 1
        return y

    def _logistic_loss_bivariate(in_tensor: Tensor, target: Tensor) -> Tensor:
        yx = in_tensor * _build_y_tensor(target)
        return -F.logsigmoid(yx).mean()

    def _sigmoid_loss_bivariate(in_tensor: Tensor, target: Tensor) -> Tensor:
        yx = in_tensor * _build_y_tensor(target)
        return torch.sigmoid(-yx).mean()

    return _logistic_loss_bivariate, _sigmoid_loss_bivariate


def construct_loader(ds: Union[TensorDataset, TextDataset], bs: int, shuffle: bool = True,
                     drop_last: bool = False) -> Union[DeviceDataLoader, Iterator]:
    r""" Construct \p Iterator which emulates a \p DataLoader """
    if isinstance(ds, TextDataset):
        return Iterator(dataset=ds, batch_size=bs, shuffle=shuffle, device=TORCH_DEVICE)

    dl = DataLoader(dataset=ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last,
                    num_workers=NUM_WORKERS, pin_memory=False)
    # noinspection PyArgumentList
    return DeviceDataLoader(dl=dl, device=TORCH_DEVICE)


def construct_filename(prefix: str, args: Namespace, out_dir: Path, file_ext: str,
                       include_loss_field: bool = True, add_timestamp: bool = False) -> Path:
    r""" Standardize naming scheme for the filename """

    def _classes_to_str(cls_set: Set[Enum]) -> str:
        return ",".join([x.name.lower() for x in sorted(cls_set)])

    fields = [prefix] if prefix else []
    if include_loss_field:
        fields.append(args.loss.name.lower())
    fields += [f"n-p={args.size_p}", f"n-n={args.size_n}", f"n-u={args.size_u}",
               f"pos={_classes_to_str(args.pos)}", f"neg={_classes_to_str(args.neg)}"]

    if args.bias:
        bias_str = ','.join([f"{x:.2f}" for _, x in args.bias])
        fields.append(f"bias={bias_str}")

    if args.preprocess:
        fields.append("preprocess")
    else:
        # Sequence length only matters if LSTM
        fields.append(f"seq={args.seq_len}")

    if add_timestamp:
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        fields.append(time_str)

    if file_ext[0] != ".": file_ext = "." + file_ext
    fields[-1] += file_ext

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "_".join(fields)
