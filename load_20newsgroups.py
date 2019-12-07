from argparse import Namespace
import copy
from collections import Counter
from dataclasses import dataclass
from enum import Enum
import itertools
import logging
import os
from pathlib import Path
import pickle as pk
from typing import List, Optional, Set, Tuple, Union

import h5py
import nltk
import nltk.tokenize
import numpy as np
import sklearn.datasets
from allennlp.commands.elmo import ElmoEmbedder
from sklearn import preprocessing
# noinspection PyProtectedMember
from sklearn.utils import Bunch

import torch
from torch import Tensor
import torchtext
from torch.utils.data import TensorDataset
from torchtext.data import Example, Field, LabelField
import torchtext.datasets
from torchtext.data.dataset import Dataset
import torchtext.vocab

from allennlp.common.file_utils import cached_path

# Valid Choices - Any subset of: ('headers', 'footers', 'quotes')
from pubn import BASE_DIR, DATA_DIR, IS_CUDA, NEG_LABEL, POS_LABEL, U_LABEL, construct_filename, \
    calculate_prior

# DATASET_REMOVE = ('headers', 'footers', 'quotes')  # ToDo settle on dataset elements to remove
DATASET_REMOVE = ()
VALID_DATA_SUBSETS = ("train", "test", "all")

DATA_COL = "data"
LABEL_COL = "target"
LABEL_NAMES_COL = "target_names"

# Validation set is disjoint from the training set.  If dataet size is n, total set size is
# n * (1 + VALIDATION_FRAC).
VALIDATION_FRAC = 0.2


class NewsgroupsCategories(Enum):
    ALT = {0}
    COMP = {1, 2, 3, 4, 5}
    MISC = {6}
    REC = {7, 8, 9, 10}
    SCI = {11, 12, 13, 14}
    SOC = {15}
    TALK = {16, 17, 18, 19}

    def __lt__(self, other: 'NewsgroupsCategories') -> bool:
        return min(self.value) < min(other.value)


@dataclass(init=True)
class NewsgroupsSerial:
    r""" Encapsulates the 20 newsgroups dataset """
    text: Field
    label: LabelField
    train: Dataset = None
    valid: Dataset = None
    test: Dataset = None
    unlabel: Dataset = None

    prior: float = None

    @staticmethod
    def _pickle_filename(args: Namespace) -> Path:
        r""" File name for pickle file """
        serialize_dir = BASE_DIR / "tensors"
        return construct_filename("data", args, serialize_dir, "pk", include_loss_field=False)

    @classmethod
    def serial_exists(cls, args: Namespace) -> bool:
        r""" Return \p True if a serialized dataset exists for the configuration in \p args """
        serial_path = cls._pickle_filename(args)
        return serial_path.exists()

    def dump(self, args: Namespace):
        r""" Serialize the newsgroup data to disk """
        path = self._pickle_filename(args)

        msg = f"Writing serialized file {str(path)}"
        flds = {k: v.examples if isinstance(v, Dataset) else v for k, v in vars(self).items()}
        logging.debug(f"Starting: {msg}")
        with open(str(path), "wb+") as f_out:
            pk.dump(flds, f_out)
        logging.debug(f"COMPLETED: {msg}")

    def build_fields(self):
        r""" Construct the dataset fields """
        return [("text", self.text), ("label", self.label)]

    @classmethod
    def load(cls, args: Namespace):
        r""" Load the serialized newsgroups dataset """
        path = cls._pickle_filename(args)

        with open(str(path), "rb") as f_in:
            flds = pk.load(f_in)
        newsgroup = cls(text=flds["text"], label=flds["label"])

        for key in vars(newsgroup).keys():
            if newsgroup.__getattribute__(key) is not None:
                continue
            newsgroup.__setattr__(key, Dataset(flds[key], newsgroup.build_fields()))
        return newsgroup


def _download_nltk_tokenizer():
    r""" NLTK uses 'punkt' tokenizer which needs to be downloaded """
    # Download the nltk tokenizer
    nltk_path = DATA_DIR / "nltk"
    nltk_path.mkdir(parents=True, exist_ok=True)
    nltk.data.path.append(str(nltk_path))
    nltk.download("punkt", download_dir=str(nltk_path))


def _download_20newsgroups(subset: str, pos_cls: Set[int], neg_cls: Set[int]):
    r"""
    Gets the specified \p subset of the 20 Newsgroups dataset.  If necessary, the dataset is
    downloaded.  It also tokenizes the imported dataset

    :param subset: Valid choices, "train", "test", and "all"
    :return: Dataset
    """
    msg = f"Download {subset} 20 newsgroups dataset"
    logging.debug(f"Starting: {msg}")

    newsgroups_dir = DATA_DIR / "20_newsgroups"
    assert not newsgroups_dir.is_file(), "Must be a directory"
    assert subset in VALID_DATA_SUBSETS, "Invalid data subset"

    newsgroups_dir.mkdir(parents=True, exist_ok=True)
    # noinspection PyUnresolvedReferences
    bunch = sklearn.datasets.fetch_20newsgroups(data_home=newsgroups_dir, shuffle=False,
                                                remove=DATASET_REMOVE, subset=subset)
    all_cls = pos_cls | neg_cls
    keep_idx = [val in all_cls for val in bunch[LABEL_COL]]
    assert any(keep_idx), "No elements to keep list"

    for key, val in bunch.items():
        if not isinstance(val, (list, np.ndarray)): continue
        if len(val) != len(keep_idx): continue
        bunch[key] = list(itertools.compress(val, keep_idx))

    logging.debug(f"COMPLETED: {msg}")
    return bunch


def _select_indexes_uar(orig_size: int, new_size: int) -> np.ndarray:
    r"""
    Selects a set of indices uniformly at random (uar) without replacement.
    :param orig_size: Original size of the array
    :param new_size: New size of the array
    :return: Boolean list of size \p original_size where \p True represents index selected
    """
    shuffled = np.arange(orig_size)
    np.random.shuffle(shuffled)
    keep_idx = np.zeros_like(shuffled, dtype=np.bool)
    for i in np.arange(new_size):
        keep_idx[shuffled[i]] = True
    return keep_idx


def _reduce_to_fixed_size(bunch: Bunch, new_size: int):
    r""" Reduce the bunch to a fixed size """
    orig_size = len(bunch[LABEL_COL])
    assert orig_size >= new_size

    keep_idx = _select_indexes_uar(orig_size, new_size)
    return _filter_bunch_by_idx(bunch, keep_idx)


def _filter_bunch_by_idx(bunch: Bunch, keep_idx: np.ndarray):
    r"""
    Filters \p Bunch object and removes any unneeded elements

    :param bunch: Dataset \p Bunch object to filter
    :param keep_idx: List of Boolean values where the value is \p True if the element should be
                     kept.
    :return: Filtered \p Bunch object
    """
    bunch = copy.deepcopy(bunch)
    for key, val in bunch.items():
        if not isinstance(val, (list, np.ndarray)): continue
        if len(keep_idx) != len(val): continue

        bunch[key] = list(itertools.compress(val, keep_idx))
        if isinstance(val, np.ndarray): bunch[key] = np.asarray(bunch[key])
    return bunch


def _configure_binary_labels(bunch: Bunch, pos_cls: Set[int], neg_cls: Set[int]):
    r""" Takes the 20 Newsgroup labels and binarizes them """
    def _is_pos(lbl: int) -> int:
        if lbl in pos_cls: return POS_LABEL
        if lbl in neg_cls: return NEG_LABEL
        raise ValueError(f"Unknown label {lbl}")

    bunch[LABEL_COL] = np.asarray(list(map(_is_pos, bunch[LABEL_COL])), dtype=np.int64)


def _convert_selected_idx_to_keep_list(sel_idx: np.ndarray, keep_list_size: int) -> np.ndarray:
    r"""
    Converts the list of integers into a binary vector with the specified indices \p True.

    :param sel_idx: Indices of the return vector to be \p True.
    :param keep_list_size: Size of the Boolean vector to return
    :return: Boolean vector with integers in \p sel_idx \p True and otherwise \p False
    """
    assert keep_list_size > max(sel_idx), "Invalid size for the keep list"
    keep_idx = np.zeros((keep_list_size,), dtype=np.bool)
    for idx in sel_idx:
        keep_idx[idx] = True
    return keep_idx


def _get_idx_of_classes(bunch: Bunch, cls_ids: Set[int]) -> np.ndarray:
    r""" Returns index of all examples in \p Bunch whose label is in \p cls_ids """
    return np.asarray([idx for idx, lbl in enumerate(bunch[LABEL_COL]) if lbl in cls_ids],
                       dtype=np.int32)


def _select_items_from_bunch(bunch: Bunch, selected_cls: Set[int], selected_idx: np.ndarray,
                             remove_sel_from_bunch: bool) -> Tuple[Bunch, Bunch]:
    r"""
    Selects a set of items (given by indices in \p selected_idx) and returns it as a new \p Bunch.
    Optionally filters the input \p bunch to remove the selected items.

    :param bunch: Bunch to select from
    :param selected_cls: Class ID numbers for the selected items
    :param selected_idx: Index of the elements in \p bunch to select
    :param remove_sel_from_bunch: If \p True, removed selected indexes from \p bunch and return it
    :return: Selected \p Bunch and other \p Bunch optionally filtered
    """
    assert len(bunch[LABEL_COL]) > max(selected_idx), "Invalid size for the keep list"

    keep_idx = _convert_selected_idx_to_keep_list(selected_idx, len(bunch[LABEL_COL]))

    sel_bunch = _filter_bunch_by_idx(bunch, keep_idx)
    sel_bunch[LABEL_NAMES_COL] = [bunch[LABEL_NAMES_COL][idx] for idx in sorted(list(selected_cls))]

    # Sanity check no unexpected classes in selected bunch
    assert all(x in selected_cls for x in sel_bunch[LABEL_COL]), "Invalid selected class in bunch"

    if remove_sel_from_bunch:
        return sel_bunch, _filter_bunch_by_idx(bunch, ~keep_idx)
    return sel_bunch, bunch


def _select_bunch_uar(size: int, bunch: Bunch, cls_ids: Set[int],
                      remove_from_bunch: bool) -> Tuple[Bunch, Bunch]:
    r"""
    Selects elements with a class label in cls_ids uniformly at random (uar) without replacement
    from \p bunch.  Optionally removes those elements from \p bunch as well.

    :param size: Number of (positive elements) to select from Bunch
    :param bunch: Source \p Bunch
    :param cls_ids: List of classes in the selected \p Bunch
    :param remove_from_bunch: If \p True, elements in the selected bunch are removed from \p bunch.
    :return: Tuple of the selected bunch and the other bunch (optionally filtered)
    """
    cls_idx = _get_idx_of_classes(bunch, cls_ids)
    sel_keep_idx = _select_indexes_uar(len(cls_idx), size)
    sel_idx = np.array(list(itertools.compress(cls_idx, sel_keep_idx)))

    return _select_items_from_bunch(bunch, cls_ids, sel_idx, remove_from_bunch)


def _select_negative_bunch(size_n: int, bunch: Bunch, neg_cls: Set[int],
                           bias: Optional[List[Tuple[NewsgroupsCategories, float]]],
                           remove_from_bunch: bool) -> Tuple[Bunch, Bunch]:
    r"""
    Randomly selects a negative bunch of size \p size_n.  If \p bias is \p None, the negative bunch
    is selected u.a.r. from all class IDs in \p neg_cls.  Otherwise, probability each group is
    selected is specified by the \p bias vector.  Optionally removes the selected elements
    from \p bunch.

    :param size_n:  Size of new negative set.
    :param bunch: Bunch from which to select the negative elements.
    :param neg_cls: ID numbers for the negative set
    :param bias: Optional vector for bias
    :param remove_from_bunch: If \p True, elements in the selected bunch are removed from \p bunch.
    :return: Tuple of the selected bunch and the other bunch (optionally filtered)
    """
    # If no bias, select the elements u.a.r.
    if bias is None:
        return _select_bunch_uar(size_n, bunch, neg_cls, remove_from_bunch)

    # Multinomial distribution from Pr[x|y=-1,s =+1]
    grp_sizes = np.random.multinomial(size_n, [prob for _, prob in bias])
    # Determine selected index
    sel_idx = []
    for (cls_lst, _), num_ele in zip(bias, grp_sizes):
        cls_idx = _get_idx_of_classes(bunch, cls_lst.value)
        assert len(cls_idx) >= num_ele, "Insufficient elements in list"
        keep_idx = _select_indexes_uar(len(cls_idx), num_ele)
        sel_idx.append(np.array(list(itertools.compress(cls_idx, keep_idx)), dtype=np.int))

    sel_idx = np.concatenate(sel_idx, axis=0)
    return _select_items_from_bunch(bunch, neg_cls, sel_idx, remove_from_bunch)


def _bunch_to_ds(bunch: Bunch, text: Field, label: LabelField) -> Dataset:
    r""" Converts the \p bunch to a classification dataset """
    fields = [('text', text), ('label', label)]
    examples = [Example.fromlist(x, fields) for x in zip(bunch[DATA_COL], bunch[LABEL_COL])]
    return Dataset(examples, fields)


def _print_stats(ngd: NewsgroupsSerial):
    r""" Log information about the dataset as a sanity check """
    logging.info(f"Maximum sequence length: {ngd.text.fix_length}")
    logging.info(f"Length of Text Vocabulary: {str(len(ngd.text.vocab))}")
    logging.info(f"Vector size of Text Vocabulary: {ngd.text.vocab.vectors.shape[1]}")
    logging.info("Label Length: " + str(len(ngd.label.vocab)))
    for k, v in vars(ngd).items():
        if not isinstance(v, Dataset): continue
        logging.info(f"{k}: Dataset Size: {len(v)}")


def _build_train_set(p_bunch: Bunch, u_bunch: Bunch, n_bunch: Optional[Bunch],
                     text: Field, label: LabelField) -> Dataset:
    r"""
    Convert the positive, negative, and unlabeled \p Bunch objects into a Dataset
    """
    data, labels, names = [], [], []
    for bunch, lbl in ((n_bunch, NEG_LABEL), (u_bunch, U_LABEL), (p_bunch, POS_LABEL)):
        if bunch is None: continue
        data.extend(bunch[DATA_COL])
        labels.append(np.full_like(bunch[LABEL_COL], lbl))

    t_bunch = copy.deepcopy(u_bunch)
    t_bunch[DATA_COL] = data
    t_bunch[LABEL_COL] = np.concatenate(labels, axis=0)
    return _bunch_to_ds(t_bunch, text, label)


def _log_category_frequency(p_cls: Set[NewsgroupsCategories], ds_name: str,
                            bunch: Union[Tensor, Bunch]) -> None:
    r"""
    Print the breakdown of classes in the \p Bunch

    :param p_cls: Categories in the positive class
    :param ds_name: Name of the dataset, e.g., "P", "N", "U", "Test"
    :param bunch: Bunch to get the class probabilities
    """
    if isinstance(bunch, Bunch):
        counter = Counter(bunch[LABEL_COL])
    else:
        assert isinstance(bunch, Tensor), "Expected tensor but unknown type"
        counter = Counter(bunch.numpy())
    tot = sum(counter.values())

    pos_sum = 0
    for cat in sorted([c for c in NewsgroupsCategories]):
        cls_sum = sum(counter[cls_id] for cls_id in cat.value)
        if cat in p_cls: pos_sum += cls_sum
        logging.debug(f"{ds_name} Class {cat.name}: {100 * cls_sum / tot:.1f}% ({cls_sum}/{tot})")
    logging.debug(f"{ds_name} Prior: {100 * pos_sum / tot:.1f}%")


def _create_serialized_20newsgroups_iterator(args):
    r"""
    Creates a serialized 20 newsgroups dataset

    :param args: Test setup information
    """
    p_cls = {cls_id for cls_grp in args.pos for cls_id in cls_grp.value}
    n_cls = {cls_id for cls_grp in args.neg for cls_id in cls_grp.value}
    complete_train = _download_20newsgroups("train", p_cls, n_cls)

    tokenizer = nltk.tokenize.word_tokenize
    # noinspection PyPep8Naming
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True,
                 fix_length=args.seq_len)
    # noinspection PyPep8Naming
    LABEL = LabelField(sequential=False)
    complete_ds = _bunch_to_ds(complete_train, TEXT, LABEL)
    cache_dir = DATA_DIR / "vector_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    TEXT.build_vocab(complete_ds, min_freq=2,
                     vectors=torchtext.vocab.GloVe(name="6B", dim=args.embed_dim, cache=cache_dir))

    size_scalar = 1 + VALIDATION_FRAC
    p_bunch, u_bunch = _select_bunch_uar(int(args.size_p * size_scalar), complete_train, p_cls,
                                         remove_from_bunch=False)
    n_bunch, u_bunch = _select_negative_bunch(int(args.size_n * size_scalar), u_bunch, n_cls,
                                              args.bias, remove_from_bunch=False)
    u_bunch = _reduce_to_fixed_size(u_bunch, new_size=int(args.size_u * size_scalar))

    test_bunch = _download_20newsgroups("test", p_cls, n_cls)

    for name, bunch in (("P", p_bunch), ("N", n_bunch), ("U", u_bunch), ("Test", test_bunch)):
        _log_category_frequency(args.pos, name, bunch)

    # Binarize the labels
    for bunch in (p_bunch, u_bunch, n_bunch, test_bunch):
        _configure_binary_labels(bunch, pos_cls=p_cls, neg_cls=n_cls)

    # Sanity check
    assert np.all(p_bunch[LABEL_COL] == POS_LABEL), "Negative example in positive (labeled) set"
    assert len(p_bunch[LABEL_COL]) == int(args.size_p * size_scalar), \
        "Positive set has wrong number of examples"
    assert np.all(n_bunch[LABEL_COL] == NEG_LABEL), "Positive example in negative (labeled) set"
    assert len(n_bunch[LABEL_COL]) == int(args.size_n * size_scalar), \
        "Negative set has wrong number of examples"
    assert len(u_bunch[LABEL_COL]) == int(args.size_u * size_scalar), \
        "Unlabeled set has wrong number of examples"

    ng_data = NewsgroupsSerial(text=TEXT, label=LABEL)
    full_train_ds = _build_train_set(p_bunch, u_bunch, n_bunch, TEXT, LABEL)
    split_ratio = 1 / (1 + VALIDATION_FRAC)
    ng_data.train, ng_data.valid = full_train_ds.split(split_ratio, stratified=True)

    ng_data.unlabel = _bunch_to_ds(u_bunch, TEXT, LABEL)
    ng_data.test = _bunch_to_ds(test_bunch, TEXT, LABEL)

    tot_unlabel_size = args.size_p + args.size_n + args.size_u
    assert len(ng_data.train.examples) == tot_unlabel_size, "Train dataset is wrong size"

    LABEL.build_vocab(ng_data.train, ng_data.test)
    ng_data.dump(args)


def _load_newsgroups_iterator(args: Namespace) -> NewsgroupsSerial:
    r"""
    Automatically downloads the 20 newsgroups dataset.
    :param args: Parsed command line arguments
    """
    assert args.pos and args.neg, "Class list empty"
    assert not (args.pos & args.neg), "Positive and negative classes not disjoint"

    # Load the serialized file if it exists
    if not NewsgroupsSerial.serial_exists(args):
        _create_serialized_20newsgroups_iterator(args)
    # _create_serialized_20newsgroups(serialize_path, args)

    serial = NewsgroupsSerial.load(args)
    serial.prior = calculate_prior(serial.test)
    if args.rho is not None:
        assert (1 - serial.prior) >= args.rho, "Input parameter rho invalid given dataset"

    _print_stats(serial)
    return serial


# ================================================================================= #
#   Preprocessed related functions
# ================================================================================= #


OPTION_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B" \
              "/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
WEIGHT_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B" \
              "/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

PREPROCESSED_FIELD = "data"


@dataclass
class NewsgroupsPreprocessed:
    prior: float = None
    train: TensorDataset = None
    valid: TensorDataset = None
    unlabel: TensorDataset = None
    test: TensorDataset = None

    @staticmethod
    def _pickle_filename(args: Namespace) -> Path:
        r""" Filename for the serialized file """
        path = BASE_DIR / "tensors" / "preprocessed"
        return construct_filename("preprocessed", args, out_dir=path, file_ext="pk",
                                  include_loss_field=False)

    @classmethod
    def serial_exists(cls, args: Namespace) -> bool:
        r""" Returns \p True if the serialized file exists """
        return cls._pickle_filename(args).exists()

    def dump(self, args: Namespace) -> None:
        r""" Serialize the newsgroup data to disk """
        path = self._pickle_filename(args)

        msg = f"Writing serialized (preprocessed) file {str(path)}"
        logging.debug(f"Starting: {msg}")
        with open(str(path), "wb+") as f_out:
            pk.dump(self, f_out)
        logging.debug(f"COMPLETED: {msg}")

    @classmethod
    def load(cls, args: Namespace):
        r""" Load the serialized (preprocessed) newsgroups dataset """
        path = cls._pickle_filename(args)
        with open(str(path), "rb") as f_in:
            return pk.load(f_in)


def _build_elmo_file_path(ds_name: str) -> Path:
    r"""
    Constructs the file path to store the preprocessed vector h5py file.
    :param ds_name: Either "test" or "train"
    :return: Path to the elmo file directory
    """
    newsgroups_dir = DATA_DIR / "20_newsgroups"
    newsgroups_dir.mkdir(parents=True, exist_ok=True)

    return newsgroups_dir / f"20newsgroups_elmo_mmm_{ds_name}.hdf5"


def _generate_preprocessed_vectors(ds_name: str, newsgroups: Bunch, path: Path) -> None:
    r"""
    Constructs the preprocessed vectors for either the test or train datasets.
    :param ds_name: Either "test" or "train"
    :param newsgroups: Scikit-Learn object containing the 20 newsgroups dataset
    :param path: Location to write serialized vectors
    """
    assert ds_name == "train" or ds_name == "test"
    n = len(newsgroups.data)

    allennlp_dir = DATA_DIR / "allennlp"
    allennlp_dir.mkdir(parents=True, exist_ok=True)
    os.putenv('ALLENNLP_CACHE_ROOT', str(allennlp_dir))

    def _make_elmo(n_device: int) -> ElmoEmbedder:
        return ElmoEmbedder(cached_path(OPTION_FILE, allennlp_dir),
                            cached_path(WEIGHT_FILE, allennlp_dir), n_device)

    # First learner use CUDA, second does not
    elmos = [_make_elmo(i) for i in range(0, -2, -1) if i < 0 or IS_CUDA]
    data = np.zeros([n, 9216])

    msg = f"Creating the preprocessed vectors for \"{ds_name}\" set"
    logging.info(f"Starting: {msg}")
    # Has to be out of for loop or stdout overwrite messes up
    if not IS_CUDA: logging.info('CUDA unavailable for ELMo encoding')
    for i in range(n):
        item = [nltk.tokenize.word_tokenize(newsgroups.data[i])]
        print(f"Processing {ds_name} document {i+1}/{n}", end="", flush=True)
        with torch.no_grad():
            try:
                em = elmos[0].embed_batch(item)
            except RuntimeError:
                em = elmos[1].embed_batch(item)
        em = np.concatenate(
                [np.mean(em[0], axis=1).flatten(),
                 np.min(em[0], axis=1).flatten(),
                 np.max(em[0], axis=1).flatten()])
        data[i] = em
        # Go back to beginning of the line. Weird formatting due to PyCharm issues
        print('\r', end="")

    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(str(path), 'w')
    f.create_dataset(PREPROCESSED_FIELD, data=data)
    f.close()
    logging.info(f"COMPLETED: {msg}")


def _binarize_tensor_labels(y: Tensor, p_cls: Set[int], n_cls: Set[int]) -> Tensor:
    r""" Binarize the labels """
    bin_idx = [torch.full_like(y, NEG_LABEL) for _ in range(2)]
    for i, cls_set in enumerate((p_cls, n_cls)):
        for cls_id in cls_set:
            bin_idx[i][y == cls_id] = POS_LABEL

    return bin_idx[0]  # First element is positive list


def _select_tensor_uar(x: Tensor, y: Tensor, cls_ids: Set[int], size: int) -> Tuple[Tensor, Tensor]:
    r""" Select a subset of the data """
    idx = []
    for i in range(y.numel()):
        if int(y[i]) in cls_ids:
            idx.append(i)
    assert len(idx) >= size, "Dataset too small"
    idx, _ = torch.tensor(idx)[torch.randperm(len(idx))][:size].sort()
    return x[idx], y[idx]


def _select_neg_tensor(x: Tensor, y: Tensor, n_cls: Set[int],
                       bias: Optional[List[Tuple[NewsgroupsCategories, float]]],
                       size_n: int) -> Tuple[Tensor, Tensor]:
    r"""
    Randomly selects a negative bunch of size \p size_n.  If \p bias is \p None, the negative bunch
    is selected u.a.r. from all class IDs in \p neg_cls.  Otherwise, probability each group is
    selected is specified by the \p bias vector.  Optionally removes the selected elements
    from \p bunch.

    :param x: (Negative) X tensor
    :param y: (Negative) y tensor
    :param n_cls: ID numbers for the negative set
    :param bias: Optional vector for bias
    :param size_n:  Size of new negative set.
    """
    # If no bias, select the elements u.a.r.
    if bias is None:
        return _select_tensor_uar(x, y, n_cls, size_n)

    # Multinomial distribution from Pr[x|y=-1,s =+1]
    grp_sizes = np.random.multinomial(size_n, [prob for _, prob in bias])
    # Determine selected index
    all_x, all_y = [], []
    for (cls_lst, _), size in zip(bias, grp_sizes):
        if size == 0:
            continue
        sub_x, sub_y = _select_tensor_uar(x, y, cls_lst.value, size)
        all_x.append(sub_x)
        all_y.append(sub_y)

    return torch.cat(all_x), torch.cat(all_y)


def _valid_split(x: Tensor) -> Tuple[Tensor, Tensor]:
    r""" Split tensor into test and validation sets """
    n = x.shape[0]
    idx = torch.randperm(n)

    split_point = int(n / (1 + VALIDATION_FRAC))
    return x[idx[:split_point]], x[idx[split_point:]]


def _create_serialized_20newsgroups_preprocessed(args: Namespace) -> None:
    r""" Serializes the 20 newsgroups as preprocessed vectors """
    p_ids = {x for cat in args.pos for x in cat.value}
    n_ids = {x for cat in args.neg for x in cat.value}

    ngp = NewsgroupsPreprocessed()
    for ds_name in ("train", "test"):
        newsgroups_dir = DATA_DIR / "20_newsgroups"
        newsgroups_dir.mkdir(parents=True, exist_ok=True)
        # shuffle=True is used since ElmoEmbedder stores states between sentences so randomness
        # should reduce this effect
        # noinspection PyUnresolvedReferences
        bunch = sklearn.datasets.fetch_20newsgroups(subset=ds_name, data_home=newsgroups_dir,
                                                    shuffle=True, remove=DATASET_REMOVE)

        path = _build_elmo_file_path(ds_name)
        if not path.exists():
            _generate_preprocessed_vectors(ds_name, bunch, path)

        vecs = h5py.File(str(path), 'r')
        x = preprocessing.scale(vecs[PREPROCESSED_FIELD][:])
        x, y = torch.from_numpy(x).float(), torch.from_numpy(bunch.target)

        _log_category_frequency(args.pos, ds_name, y)
        fld_name = "unlabel" if ds_name == "train" else ds_name
        ngp.__setattr__(fld_name, TensorDataset(x, _binarize_tensor_labels(y, p_ids, n_ids)))

        if ds_name == "train":
            unlabel_x, unlabel_y = x, y
        elif ds_name == "test":
            p_mask = (_binarize_tensor_labels(y, p_ids, n_ids) == POS_LABEL)
            ngp.prior = y[p_mask].numel() / y.numel()

    scale = 1 + VALIDATION_FRAC
    # noinspection PyUnboundLocalVariable,PyUnboundLocalVariable
    p_tens = _select_tensor_uar(unlabel_x, unlabel_y, p_ids, int(scale * args.size_p))
    u_tens = _select_tensor_uar(unlabel_x, unlabel_y, p_ids.union(n_ids), int(scale * args.size_u))
    n_tens = _select_neg_tensor(unlabel_x, unlabel_y, n_ids, args.bias, int(scale * args.size_n))

    # Sanity check the distribution
    _log_category_frequency(args.pos, "Pos", p_tens[1])
    _log_category_frequency(args.pos, "Unlabel", u_tens[1])
    _log_category_frequency(args.pos, "Neg", n_tens[1])
    logging.debug(f"Test Prior: {100 * ngp.prior:.2f}%%")

    full_x = [_valid_split(x) for x, _ in (p_tens, u_tens, n_tens)]
    full_y = [POS_LABEL, U_LABEL, NEG_LABEL]

    # Build the validation and train set
    for idx, (name, scale) in enumerate((("train", 1), ("valid", VALIDATION_FRAC))):
        x_grp = [split[idx] for split in full_x]
        y_grp = [torch.full(x.shape[:1], lbl, dtype=torch.int64) for x, lbl in zip(x_grp, full_y)]

        x_tensor, y_tensor = torch.cat(x_grp, dim=0), torch.cat(y_grp, dim=0)
        assert x_tensor.shape[0] == y_tensor.shape[0], "Tensor shape mismatch"
        ngp.__setattr__(name, TensorDataset(x_tensor, y_tensor))
    ngp.dump(args)


def load(args: Namespace):
    assert args.pos and args.neg, "Class list empty"
    assert not (args.pos & args.neg), "Positive and negative classes not disjoint"

    _download_nltk_tokenizer()

    if not args.preprocess:
        return _load_newsgroups_iterator(args)

    if not NewsgroupsPreprocessed.serial_exists(args):
        _create_serialized_20newsgroups_preprocessed(args)
    return NewsgroupsPreprocessed.load(args)
