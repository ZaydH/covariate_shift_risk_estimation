from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
import logging
from typing import ClassVar, Optional

import numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score

from fastai.metrics import auc_roc_score
import torch
from torch import Tensor
from torchtext.data import Dataset, LabelField

from pubn import BASE_DIR, POS_LABEL, construct_filename, construct_loader, \
    get_forward_input_and_labels
from pubn.model import NlpBiasedLearner


@dataclass
class LearnerResults:
    r""" Encapsulates ALL results for a single NLP learner MODEL """
    FIELD_SEP: ClassVar[str] = ","

    @dataclass(init=True)
    class DatasetResult:
        r""" Encapsulates results of a model on a SINGLE DATASET """
        ds_size: int
        accuracy: float = None
        auroc: float = None
        auprc: float = None
        f1: float = None

    valid_loss = None
    unlabel = None
    test = None


def calculate_results(args: Namespace, classifier: NlpBiasedLearner, labels: Optional[LabelField],
                      unlabel_ds: Dataset, test_ds: Dataset):
    r""" Calculates and writes to disk the model's results """
    classifier.eval()

    res = LearnerResults()
    res.valid_loss = classifier.best_loss

    for ds, name in ((unlabel_ds, "unlabel"), (test_ds, "test")):
        itr = construct_loader(ds, bs=args.bs, shuffle=False)
        all_y, dec_scores = [], []
        with torch.no_grad():
            for batch in itr:
                forward_in, lbls = get_forward_input_and_labels(batch)
                all_y.append(lbls)
                dec_scores.append(classifier.forward(*forward_in))

        # Iterator transforms label so transform it back
        y = tfm_y = torch.cat(all_y, dim=0).squeeze()
        if labels is not None:
            y = torch.full_like(tfm_y, -1)
            y[tfm_y == labels.vocab.stoi[POS_LABEL]] = 1
        y = y.cpu().numpy()

        dec_scores = torch.cat(dec_scores, dim=0).squeeze()
        y_hat, dec_scores = dec_scores.sign().cpu().numpy(), dec_scores.cpu().numpy()

        res.__setattr__(name, _single_ds_results(name, args, y, y_hat, dec_scores))

    _write_results_to_disk(args, res)


def _single_ds_results(ds_name: str, args: Namespace, y: np.ndarray, y_hat: np.ndarray,
                       dec_scores: np.ndarray) -> LearnerResults.DatasetResult:
    r""" Logs and returns the results on a single dataset """
    loss_name = args.loss.name
    results = LearnerResults.DatasetResult(y.shape[0])

    str_prefix = f"{loss_name} {ds_name}:"

    logging.debug(f"{str_prefix} Dataset Size: %d", results.ds_size)
    # Pre-calculate fields needed in other calculations
    results.conf_matrix = confusion_matrix(y, y_hat)
    assert np.sum(results.conf_matrix) == results.ds_size, "Verify size matches"

    # Calculate prior information
    results.accuracy = np.trace(results.conf_matrix) / results.ds_size
    logging.debug(f"{str_prefix} Accuracy = %.3f%%", 100. * results.accuracy)

    results.auroc = auc_roc_score(torch.tensor(dec_scores).cpu(), torch.tensor(y).cpu())
    logging.debug(f"{str_prefix} AUROC: %.6f", results.auroc)

    results.auprc = average_precision_score(y, dec_scores)
    logging.debug(f"{str_prefix} AUPRC %.6f", results.auprc)

    results.f1 = float(f1_score(y, y_hat))
    logging.debug(f"{str_prefix} F1-Score: %.6f", results.f1)

    logging.debug(f"{str_prefix} Confusion Matrix:\n{results.conf_matrix}")
    results.conf_matrix = str(results.conf_matrix).replace("\n", " ")

    return results


def _write_results_to_disk(args: Namespace, res: LearnerResults) -> None:
    r""" Logs the results to disk for later analysis """
    def _log_val(_v) -> str:
        if isinstance(_v, str): return _v
        if isinstance(_v, bool): return str(_v)
        if isinstance(_v, int): return f"{_v:d}"
        if isinstance(_v, Tensor): _v = float(_v.item())
        if isinstance(_v, float): return f"{_v:.15f}"
        if isinstance(_v, Enum): return _v.name
        if isinstance(_v, set):
            return "|".join([_log_val(_x) for _x in sorted(_v)])
        if _v is None: return "NA"
        raise ValueError(f"Unknown value type \"{type(_v)}\" to log")

    header, fields = [], []
    for key, val in vars(args).items():
        header.append(key.replace("_", "-"))
        if key == "bias" and isinstance(val, list):
            fields.append("|".join([f"{x:.2f}" for _, x in val]))
            continue
        fields.append(_log_val(val))

    header.append("valid_loss")
    fields.append(_log_val(res.valid_loss))

    for res_name in ("unlabel", "test"):
        res_val = res.__getattribute__(res_name)
        for fld_name, fld_val in vars(res_val).items():
            header.append(f"{res_name}-{fld_name}")
            fields.append(_log_val(fld_val))

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = construct_filename(prefix="res", args=args, out_dir=results_dir, file_ext="csv",
                                  include_loss_field=True, add_timestamp=True)
    with open(str(filename), "w+") as f_out:
        f_out.write(LearnerResults.FIELD_SEP.join(header))
        f_out.write("\n")
        f_out.write(LearnerResults.FIELD_SEP.join(fields))
