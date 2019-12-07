from argparse import Namespace
import logging
from functools import partial
import math
from pathlib import Path
import time
from typing import Callable, Optional, Set, Tuple, Union

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Subset, TensorDataset
from torchtext.data import Dataset as TextDataset, Batch
from torchtext.data import LabelField, Example

from ._base_classifier import BaseClassifier, ClassifierConfig
from ._utils import BASE_DIR, DatasetType, LoaderType, NEG_LABEL, POS_LABEL, TORCH_DEVICE, \
    U_LABEL, build_loss_functions, construct_filename, construct_loader

from .logger import TrainingLogger, create_stdout_handler
from .loss import LossType, PULoss, PUbN


class NlpBiasedLearner(nn.Module):
    Config = ClassifierConfig
    _log = None

    def __init__(self, args: Namespace, embedding_weights: Optional[Tensor], prior: float):
        super().__init__()
        self._setup_logger()

        self._log.debug(f"NLP Learner: Prior: {prior:.3f}")

        self._is_rnn = (embedding_weights is not None)  # True if just a FF classifier
        self._model = BaseClassifier(embedding_weights)

        self._args = args
        self.l_type = args.loss

        self.prior = prior
        self._rho = args.rho
        self._tau = args.tau
        self._eta = None
        if self._is_pubn():
            if self._rho is None: raise ValueError("rho required for PUbN loss")
            if self._tau is None: raise ValueError("tau required for PUbN loss")
            self._sigma = SigmaLearner(embedding_weights)
        else:
            if self._rho is not None: raise ValueError("rho specified but PUbN loss not used")
            if self._tau is not None: raise ValueError("tau specified but PUbN loss not used")
            self._sigma = None

        self._prefix = self.l_type.name.lower()
        self._train_start = self._optim = self.best_loss = self._logger = None

        # True labels get mapped to different values by LabelField object.  Stores the mapped values
        self._map_pos = self._map_neg = None

        self.to(TORCH_DEVICE)

    def _configure_fit_vars(self):
        r""" Set initial values/construct all variables used in a fit method """
        # Fields that apply regardless of loss method
        self._train_start = time.time()
        self.best_loss = np.inf

        tb_dir = BASE_DIR / "tb"
        TrainingLogger.create_tensorboard(tb_dir)
        self._logger = TrainingLogger(["Train Loss", "Valid Loss", "Best", "Time"],
                                      [20, 20, 7, 10],
                                      logger_name=self.Config.LOGGER_NAME,
                                      tb_grp_name=self.l_type.name)

    def fit(self, train: DatasetType, valid: DatasetType, unlabel: DatasetType,
            label: Optional[LabelField]):
        r""" Fits the learner"""
        # noinspection PyUnresolvedReferences
        assert (label is None) ^ (self._model.is_rnn()), "label and model version mismatch"

        # Filter the dataset based on the training configuration
        if not self._is_pubn():
            exclude_lbl = NEG_LABEL if self._is_nnpu() else U_LABEL
            train = exclude_label_in_dataset(train, exclude_lbl)
            valid = exclude_label_in_dataset(valid, exclude_lbl)

        self._map_pos, self._map_neg = POS_LABEL, NEG_LABEL
        if self._is_rnn:
            self._map_pos, self._map_neg = label.vocab.stoi[POS_LABEL], label.vocab.stoi[NEG_LABEL]

        if self._is_pubn():
            self._configure_fit_vars()
            # noinspection PyUnresolvedReferences
            train_itr = construct_loader(train, bs=self._sigma.Config.BATCH_SIZE, shuffle=True)
            # noinspection PyUnresolvedReferences
            valid_itr = construct_loader(valid, bs=self._sigma.Config.BATCH_SIZE, shuffle=True,
                                         drop_last=False)
            self._fit_sigma(train_itr, valid_itr)
            # noinspection PyUnresolvedReferences
            u_itr = construct_loader(unlabel, bs=self._sigma.Config.BATCH_SIZE, shuffle=True,
                                     drop_last=False)
            self._calculate_eta(u_itr)

        self._configure_fit_vars()
        # noinspection PyUnresolvedReferences
        train_itr = construct_loader(train, bs=self.Config.BATCH_SIZE, shuffle=True)
        # noinspection PyUnresolvedReferences
        valid_itr = construct_loader(valid, bs=self.Config.BATCH_SIZE, shuffle=True,
                                     drop_last=False)
        self._fit_base(train_itr, valid_itr)

    def _fit_base(self, train: LoaderType, valid: LoaderType):
        r""" Shared functions for nnPU and supervised learning """
        # noinspection PyUnresolvedReferences
        self._optim = optim.AdamW(self._model.parameters(), lr=self._model.Config.LEARNING_RATE,
                                  weight_decay=self._model.Config.WEIGHT_DECAY, amsgrad=True)

        univar_log_loss, univar_sigmoid_loss = build_loss_functions()
        bivar_log_loss, bivar_sigmoid_loss = build_loss_functions(pos_classes=self._map_pos)
        if self._is_nnpu():
            nnpu = PULoss(prior=self.prior, pos_label=self._map_pos,
                          train_loss=univar_log_loss, valid_loss=univar_sigmoid_loss)
            valid_loss = partial(nnpu.calc_valid_loss)
        elif self._is_pubn():
            pubn = PUbN(prior=self.prior, rho=self._rho, eta=self._eta,
                        pos_label=self._map_pos, neg_label=self._map_neg,
                        train_loss=univar_log_loss, valid_loss=univar_sigmoid_loss)
            valid_loss = partial(pubn.calc_valid_loss)
        else:
            assert self.l_type == LossType.PN, "Unknown loss type"
            loss_func = bivar_log_loss
            valid_loss = bivar_sigmoid_loss

        # noinspection PyUnresolvedReferences
        forward = partial(self._model.forward)
        # noinspection PyUnresolvedReferences
        for ep in range(1, self._model.Config.NUM_EPOCH + 1):
            # noinspection PyUnresolvedReferences
            self._model.train()
            if self._sigma is not None: self._sigma.eval()  # Sigma frozen after first stage

            train_loss, num_batch = torch.zeros(()), 0
            for batch in train:
                self._optim.zero_grad()

                forward_in, labels = get_forward_input_and_labels(batch)
                # noinspection PyUnresolvedReferences
                dec_scores = self._model.forward(*forward_in)

                if self._is_nnpu():
                    # noinspection PyUnboundLocalVariable
                    loss = nnpu.calc_loss(dec_scores, labels)
                    loss.grad_var.backward()
                    loss = loss.loss_var
                else:
                    if self._is_pubn():
                        sigma_x = self._sigma.forward(*forward_in)
                        # noinspection PyUnboundLocalVariable
                        loss = pubn.calc_loss(dec_scores, labels, sigma_x)
                    else:
                        assert self.l_type == LossType.PN, "Unknown loss type"
                        # noinspection PyUnboundLocalVariable
                        loss = loss_func(dec_scores, labels)
                    loss.backward()
                train_loss += loss.detach()
                num_batch += 1

                self._optim.step()
            train_loss /= num_batch
            self._log_epoch(ep, train_loss, forward, valid, valid_loss)
        self._restore_best_model()

    def _fit_sigma(self, train: LoaderType, valid: LoaderType) -> None:
        r""" Fit the sigma function Pr[s = + 1 | x] """
        self._sigma.is_fit = True
        # noinspection PyUnresolvedReferences
        self._optim = optim.AdamW(self._sigma.parameters(),
                                  lr=self._sigma.Config.LEARNING_RATE,
                                  weight_decay=self._sigma.Config.WEIGHT_DECAY, amsgrad=True)

        pos_label = {self._map_neg, self._map_pos}

        univar_log_loss, univar_sigmoid_loss = build_loss_functions()
        pu_loss = PULoss(prior=self.prior + self._rho, pos_label=pos_label,
                         train_loss=univar_log_loss, valid_loss=univar_sigmoid_loss)
        valid_loss = partial(pu_loss.calc_valid_loss)
        forward = partial(self._sigma.forward_fit)
        for ep in range(1, self.Config.NUM_EPOCH + 1):
            self._sigma.train()
            train_loss, num_batch = torch.zeros(()), 0
            for batch in train:
                self._optim.zero_grad()

                forward_in, labels = get_forward_input_and_labels(batch)
                # noinspection PyUnresolvedReferences
                dec_scores = self._sigma.forward_fit(*forward_in)

                loss = pu_loss.calc_loss(dec_scores, labels)
                loss.grad_var.backward()
                train_loss += loss.loss_var.detach()
                num_batch += 1

                self._optim.step()
            train_loss /= num_batch

            self._sigma.eval()
            self._log_epoch(ep, train_loss, forward, valid, valid_loss)
        self._restore_best_model()
        self._sigma.eval()

        self._sigma.is_fit = False

    def _calculate_eta(self, unlabel: LoaderType) -> float:
        r"""
        Calculates eta for PUbN

        :param unlabel: Set of unlabeled examples
        :return: Value of eta
        """
        sigma_x = []
        for batch in unlabel:
            forward_in, _ = get_forward_input_and_labels(batch)
            sigma_x.append(self._sigma.forward(*forward_in))
        sigma_x, _ = torch.cat(sigma_x, dim=0).squeeze().sort()

        idx = math.floor(self._tau * (1 - self.prior - self._rho) * sigma_x.numel())
        self._eta = float(sigma_x[int(idx)].item())

        return self._eta

    def _log_epoch(self, ep: int, train_loss: Tensor, forward: Callable, valid_itr: LoaderType,
                   loss_func: Callable) -> None:
        r"""
        Log the results of the epoch

        :param ep: Epoch number
        :param train_loss: Training loss value
        :param forward: \p forward method used to calculate the loss
        :param valid_itr: Validation loader (e.g., \p DataDataLoader, \p Iterator)
        :param loss_func: Function used to calculate the loss
        """
        with torch.no_grad():
            valid_loss = self._calc_valid_loss(forward, valid_itr, loss_func)

        is_best = float(valid_loss.item()) < self.best_loss
        if is_best:
            self.best_loss = float(valid_loss.item())
            save_module(self, self._build_serialize_name(self._prefix))
        self._logger.log(ep, [train_loss, valid_loss, is_best, time.time() - self._train_start])

    def _calc_valid_loss(self, forward: Callable, itr: LoaderType, loss_func: Callable) -> Tensor:
        r""" Calculate the validation loss for \p itr using forward method """
        dec_scores, labels, sigma_x = [], [], []
        with torch.no_grad():
            for batch in itr:
                forward_in, lbls = get_forward_input_and_labels(batch)
                dec_scores.append(forward(*forward_in))
                labels.append(lbls)
                if self._is_pubn() and not self._sigma.is_fit:
                    sigma_x.append(self._sigma.forward(*forward_in))
        dec_scores, labels = torch.cat(dec_scores, dim=0), torch.cat(labels, dim=0)

        if not self._is_pubn() or self._sigma.is_fit:
            return loss_func(dec_scores, labels)

        sigma_x = torch.cat(sigma_x, dim=0)
        return loss_func(dec_scores, labels, sigma_x)

    def forward(self, x: Tensor, x_len: Optional[Tensor]) -> Tensor:
        # noinspection PyUnresolvedReferences
        return self._model.forward(x, x_len).squeeze()

    def _restore_best_model(self):
        r""" Restores the best trained model from disk """
        msg = f"Restoring {self.l_type.name} best trained model"
        self._log.debug(f"Starting: {msg}")
        load_module(self, self._build_serialize_name(self._prefix))
        self._log.debug(f"COMPLETED: {msg}")
        self.eval()

    def _build_serialize_name(self, prefix: str) -> Path:
        r"""

        :param prefix: Prefix given to the name of the serialized file
        :return: \p Path to the serialized file
        """
        serialize_dir = BASE_DIR / "models"
        serialize_dir.mkdir(parents=True, exist_ok=True)
        return construct_filename(prefix, self._args, serialize_dir, "pth", add_timestamp=False)

    @classmethod
    def _setup_logger(cls) -> None:
        r""" Creates a logger for just the NLP class """
        if cls._log is not None: return
        cls._log = logging.getLogger(cls.Config.LOGGER_NAME)
        cls._log.propagate = False  # Do not propagate log messages to a parent logger
        create_stdout_handler(cls.Config.LOG_LEVEL, logger_name=cls.Config.LOGGER_NAME)

    def _is_pubn(self) -> bool:
        r""" Returns \p True if the loss is PUbN """
        return self.l_type == LossType.PUBN

    def _is_nnpu(self) -> bool:
        r""" Returns \p True if the loss is nnPU """
        return self.l_type == LossType.NNPU


class SigmaLearner(nn.Module):
    r""" Encapsulates the sigma learner """
    Config = ClassifierConfig

    def __init__(self, embedding_weights: Tensor):
        super().__init__()
        self._model = BaseClassifier(embed=embedding_weights)

        self.is_fit = False
        self.to(TORCH_DEVICE)

    def forward_fit(self, x: Tensor, x_len: Optional[Tensor]) -> Tensor:
        r""" Forward method only used during training """
        # noinspection PyUnresolvedReferences
        return self._model.forward(x, x_len)

    def forward(self, x: Tensor, x_len: Optional[Tensor]) -> Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward_fit(x, x_len))


def save_module(module: nn.Module, filepath: Path) -> None:
    r""" Save the specified \p model to disk """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # noinspection PyUnresolvedReferences
    torch.save(module.state_dict(), str(filepath))


def load_module(module: nn.Module, filepath: Path):
    r"""
    Loads the specified model in file \p filepath into \p module and then returns \p module.

    :param module: \p Module where the module on disk will be loaded
    :param filepath: File where the \p Module is stored
    :return: Loaded model
    """
    # Map location allows for mapping model trained on any device to be loaded
    module.load_state_dict(torch.load(str(filepath), map_location=TORCH_DEVICE))

    module.eval()
    return module


def exclude_label_in_dataset(ds: DatasetType,
                             label_to_exclude: Union[int, Set[int]]) -> Dataset:
    r""" Creates a new \p TextDataset that will exclude the label of \p label_to_exclude """
    if isinstance(label_to_exclude, int):
        label_to_exclude = {label_to_exclude}

    # TensorDataset allows for use of the Subset wrapper
    if isinstance(ds, TensorDataset):
        _, y = ds.tensors
        idx = [idx for idx, _y in enumerate(y) if int(_y) not in label_to_exclude]
        return Subset(ds, idx)

    def _filter_label(x: Example):
        # noinspection PyUnresolvedReferences
        return x.label not in label_to_exclude

    return TextDataset(examples=ds.examples, fields=ds.fields, filter_pred=_filter_label)


def get_forward_input_and_labels(batch: Union[Batch, Tuple[Tensor, Tensor]]) \
        -> Tuple[Tuple[Tensor, Optional[Tensor]], Tensor]:
    r"""
    Gets input to the \p forward method and the labels

    :param batch: Batch from \p Iterator/\p DeviceDataLoader
    :return: Tuple of the input to forward method and labels tensor
    """
    if isinstance(batch, Batch):
        # noinspection PyUnresolvedReferences
        return batch.text, batch.label

    assert isinstance(batch, (list, tuple)), "Unknown inputs type"
    return (batch[0], None), batch[1]
