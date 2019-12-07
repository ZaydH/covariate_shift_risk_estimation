import collections
from enum import Enum
from typing import Callable, Optional, Set, Union

import torch
from torch import Tensor
import torch.nn as nn

from ._utils import TORCH_DEVICE


class PULoss:
    """wrapper of loss function for PU learning"""

    LossInfo = collections.namedtuple("LossInfo", ["loss_var", "grad_var"])

    class Config:
        GAMMA = 1
        BETA = 0

        TAU = 0.5

    def __init__(self, prior: float, pos_label: Union[Set[int], int],
                 train_loss: Callable, use_nnpu: bool = True,
                 valid_loss: Optional[Callable] = None):
        r"""
        :param prior: Positive class prior probability, i.e., :math:`\Pr[y = +1]`
        :param pos_label: Integer labels assigned to positive-valued examples.
        :param train_loss: Loss function underlying the classifier
        :param use_nnpu: If \p True, use nnPU loss.  Otherwise, use uPU.
        :param valid_loss: Optional validation loss.  If not specified, uses \p train_loss for
                           validation.
        """
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        if isinstance(pos_label, int):
            pos_label = {pos_label}
        self.pos_label = pos_label

        self.gamma = self.Config.GAMMA
        self.beta = self.Config.BETA
        self._is_nnpu = use_nnpu

        self._train_loss = train_loss
        self._valid_loss = valid_loss if valid_loss is not None else self._train_loss

    @property
    def is_nnpu(self) -> bool:
        r"""
        Returns \p True if the loss is the nnPU loss.  Otherwise if it is the uPU loss, it returns
        \p False.
        """
        return self._is_nnpu

    def name(self) -> str:
        r""" Name of the loss, either "nnPU" or "uPU" """
        return "nnPU" if self.is_nnpu else "uPU"

    @staticmethod
    def _is_torch(tensor: Tensor) -> bool:
        r""" Return \p True if \p tensor is a \p torch tensor """
        return isinstance(tensor, Tensor)

    @classmethod
    def _verify_loss_inputs(cls, dec_scores: Tensor, labels: Tensor) -> None:
        r""" Sanity check the inputs """
        assert cls._is_torch(dec_scores), "dec_scores must be torch tensor"
        assert cls._is_torch(labels), "labels must be torch tensor"

        assert len(dec_scores.shape) == 1, "dec_scores should be a vector"
        assert len(labels.shape) == 1, "labels should be a vector"
        assert dec_scores.shape[0] == labels.shape[0], "Batch size mismatch"

        assert dec_scores.dtype == torch.float, "dec_scores tensor must be float"
        assert labels.dtype == torch.int64, "labels must be integers"

    def calc_loss(self, dec_scores: Tensor, label: Tensor) -> 'LossInfo':
        r"""
        nnPU uses separate approaches for determining the loss and variable used for calculating
        the gradient.
        :param dec_scores: Decision function value
        :param label: Labels for each sample in \p.
        :return: Named tuple with value used for loss and the one used for the gradient
        """
        return self._base_calc_loss(self._train_loss, dec_scores, label)

    def calc_valid_loss(self, dec_scores: Tensor, label: Tensor) -> 'LossInfo':
        r""" Calculates only the loss information """
        return self._base_calc_loss(self._valid_loss, dec_scores=dec_scores, label=label).loss_var

    def _base_calc_loss(self, loss_func: Callable, dec_scores: Tensor, label: Tensor) -> 'LossInfo':
        r"""
        nnPU uses separate approaches for determining the loss and variable used for calculating
        the gradient.
        :param loss_func: Surrogate loss function to use in the calculation
        :param dec_scores: Decision function value
        :param label: Labels for each sample in \p.
        :return: Named tuple with value used for loss and the one used for the gradient
        """
        dec_scores, label = dec_scores.squeeze(), label.squeeze()
        self._verify_loss_inputs(dec_scores, label)

        # Mask used to filter the dec_scores tensor and in loss calculations
        p_mask = self._find_p_mask(label)
        u_mask = ~p_mask
        has_p, has_u = p_mask.any(), u_mask.any()

        y_unlabel = loss_func(-dec_scores)
        neg_risk = y_unlabel[u_mask].mean() if has_u else torch.zeros((), device=TORCH_DEVICE)
        if has_p:
            y_pos = loss_func(dec_scores[p_mask])
            pos_risk = self.prior * y_pos.mean()

            neg_risk -= self.prior * y_unlabel[p_mask].mean()
        else:
            pos_risk = torch.zeros((), device=TORCH_DEVICE)  # Needs to be have len(shape) == 0

        loss = gradient_var = pos_risk + neg_risk
        if self.is_nnpu and neg_risk < -self.beta:
            loss = pos_risk - self.beta
            gradient_var = -self.gamma * neg_risk
        return self.LossInfo(loss_var=loss, grad_var=gradient_var)

    def _find_p_mask(self, labels: Tensor) -> Tensor:
        r"""
        Constructs a Boolean vector where an element is \p True if the corresponding example in
        \p labels is also \p True.
        """
        p_mask = torch.zeros(labels.shape, dtype=torch.bool, device=TORCH_DEVICE)
        for p_lbl in self.pos_label:
            p_mask |= labels == p_lbl
        return p_mask


class PUbN:
    r"""
    Positive, unlabeled, and biased negative risk estimator from:

    Yu-Guan Hsieh, Gang Niu, and Masashi Sugiyama. Classification from Positive, Unlabeled and
    Biased Negative Data. ICML 2019.
    """
    def __init__(self, prior: float, rho: float, eta: float,
                 pos_label: int, neg_label: int, train_loss: Callable,
                 valid_loss: Optional[Callable] = None):
        r"""
        :param prior: Positive class prior probability, i.e., :math:`\Pr[y = +1]`
        :param pos_label: Integer labels assigned to positive-valued examples.
        :param train_loss: Loss function underlying the classifier
        :param valid_loss: Optional validation loss.  If not specified, uses \p train_loss for
                           validaiton.
        """
        for name, val in (("prior", prior), ("eta", eta)):
            if val <= 0 or val >= 1:
                raise ValueError(f"Value of {val} must be in range (0,1)")
        if rho <= 0 or rho > 1 - prior:
            raise ValueError("rho must be in range (0, 1 - prior]")

        self.prior = prior
        self.rho = rho
        self.eta = eta

        assert pos_label != neg_label, "Positive and negative labels must be different"
        self._pos_label, self._neg_label = pos_label, neg_label

        self._train_loss = train_loss
        self._valid_loss = valid_loss if valid_loss is not None else self._train_loss

    def calc_loss(self, dec_scores: Tensor, labels: Tensor, sigma_x: Tensor) -> Tensor:
        r"""
        Calculates the positive-unlabeled biased negative loss

        :param dec_scores: Decision function scores, i.e., :math:`g(x)`
        :param labels: Label vector
        :param sigma_x: Estimated value of :math:`\sigma(x)=\Pr[s = +1 \vert x]`.
        :return: PUbN loss (see Eq. (7)) in paper
        """
        return self._base_calc_loss(self._train_loss, dec_scores, labels, sigma_x)

    def calc_valid_loss(self, dec_scores: Tensor, labels: Tensor, sigma_x: Tensor) -> Tensor:
        r"""
        Calculates the positive-unlabeled biased negative loss

        :param dec_scores: Decision function scores, i.e., :math:`g(x)`
        :param labels: Label vector
        :param sigma_x: Estimated value of :math:`\sigma(x)=\Pr[s = +1 \vert x]`.
        :return: PUbN loss (see Eq. (7)) in paper
        """
        return self._base_calc_loss(self._valid_loss, dec_scores, labels, sigma_x)

    def _base_calc_loss(self, loss_func: Callable, dec_scores: Tensor, labels: Tensor,
                        sigma_x: Tensor) -> Tensor:
        r"""
        Calculates the positive-unlabeled biased negative loss

        :param loss_func: Surrogate loss function to use in the calculation
        :param dec_scores: Decision function scores, i.e., :math:`g(x)`
        :param labels: Label vector
        :param sigma_x: Estimated value of :math:`\sigma(x)=\Pr[s = +1 \vert x]`.
        :return: PUbN loss (see Eq. (7)) in paper
        """
        # Labeled loss terms
        p_mask, bn_mask = (labels == self._pos_label), (labels == self._neg_label)
        assert not (p_mask & bn_mask).any(), "Labels not disjoint"
        u_mask = p_mask.logical_xor(bn_mask).logical_not()

        has_p, has_bn = p_mask.any(), bn_mask.any()
        # Multiplicative factors included in final sum
        l_pos = loss_func(dec_scores[p_mask]).mean() if has_p else torch.zeros(()).to(TORCH_DEVICE)
        if has_bn:
            l_bn = loss_func(-dec_scores[bn_mask]).mean()
        else:
            l_bn = torch.zeros((), device=TORCH_DEVICE)

        l_u_n = self._u_n_loss(loss_func, u_mask, sigma_x, dec_scores, is_u=True) \
                + self.prior * self._u_n_loss(loss_func, p_mask, sigma_x, dec_scores, is_u=False) \
                + self.rho * self._u_n_loss(loss_func, bn_mask, sigma_x, dec_scores, is_u=False)

        return self.prior * l_pos + self.rho * l_bn + l_u_n

    def _u_n_loss(self, loss_func: Callable, orig_mask: Tensor, sigma_x: Tensor, dec_scores: Tensor,
                  is_u: bool) -> Tensor:
        r"""
        Calculates a single term of the expected
        :param loss_func: Surrogate loss function to use in the calculation
        :param orig_mask: Masks for either
        :param sigma_x: Estimated value of :math:`\sigma(x)=\Pr[s = +1 \vert x]`.
        :param dec_scores: Decision function scores
        :param is_u: If \p True, using the unlabeled set
        :return: Single term in the unlabeled negative set
        """
        sigma_mask = sigma_x > self.eta
        if is_u:
            sigma_mask = ~sigma_mask
        mask = orig_mask & sigma_mask

        # No elements of a given type return 0
        if not mask.any(): return torch.zeros((), device=TORCH_DEVICE)

        dec_scores, sigma_x = dec_scores[mask], sigma_x[mask]
        loss = loss_func(-dec_scores)
        neg_sigma = -sigma_x + 1
        if not is_u:
            neg_sigma = neg_sigma / sigma_x

        return (loss * neg_sigma).sum() / orig_mask.sum(dtype=torch.int64)


class LossType(Enum):
    r""" Loss type to train the learner """
    PN = nn.BCELoss
    NNPU = PULoss
    PUBN = PUbN
