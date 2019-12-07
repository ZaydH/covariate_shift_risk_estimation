# Negative Class Covariate Shift on 20 Newsgroups

[![docs](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ZaydH/covariate_shift_risk_estimation/blob/master/LICENSE)

Compares three risk estimators on the 20 newsgroups dataset under negative covariate shift.  The supported risk estimators are:

* Positive-Negative (PN) -- Standard binary, supervised learning
* Positive-Unlabeled (PU) -- Uses the non-negative PU (nnPU) risk estimator proposed by Kiryo et al. [[1]](#1)
* Positive-Unlabeled biased Negative (PUbN) -- Proposed by Hsieh et al. [[2]](#2)

## Running the Program

The model and risk estimator definitions are in the `pubn` modules.  Users should interact with the program via the `driver.py` file.

    python driver.py size_p size_n size_u loss --pos [CAT1 ...] --neg [CAT2 ...]

* `size_p` -- Size of the labeled *positive* set
* `size_n` -- Size of the labeled *negative* set
* `size_u` -- Size of the unlabeled set
* `loss` -- Risk estimator. Valid choices are: `pn`, `nnpu`, or `pubn`
* `--pos` -- 20 newsgroups categories to use as the *positive* class. Valid choices are: `alt`, `comp`, `misc`, `rec`, `sci`, `soc`, and `talk`. Required with optionally multiple categories separated by spaces.
* `--neg` -- 20 newsgroups categories to use as the *negative* class. Same set of valid choices as `--pos` but must be disjoint from `--pos`. Required.
* `--bias` -- Optional bias vector for the categories in `--neg`.  One-to-one mapping that must be non-negative and sum to 1.
* `--rho` -- Labeling frequency.  `--bias` and `--rho` always specified together
* `--lstm` -- Train the LSTM network from scratch.  Yields significantly worse results but eliminates the need to preprocess the documents using ELMo.

For checkout purposes, we recommend calling:

    python driver.py 500 500 6000 pubn --pos alt comp misc rec --neg sci soc talk --bias 0.1 0.5 0.4 --rho 0.1 --lstm

## Dataset

The 20 newsgroups dataset is used for all experiments.  It will be automatically downloaded using the `sklearn` library.  If you are using ELMo preprocessed vectors, it may take significant time to encode the 20 newsgroups documents.  The encoded documents are serialized so this only needs to be done once. 

## CUDA Support

The implementation supports both CPU and CUDA (i.e., GPU) execution.  If CUDA is detected on the system, the implementation defaults to CUDA support.

## Requirements

This program was tested with Python 3.6.5 and 3.7.1 on MacOS and on Debian Linux.  `requirements.txt` enumerates the exact packages used. A summary of the key requirements is below.

* PyTorch (`torch`) -- 1.3.1
* PyTorch's Text Library (`torchtext`) -- 0.4.0
* Natural Language Toolkit (`nltk`) -- 3.4.5
* Allen AI Institute's NLP Library (`allennlp`) -- 0.9.0
* FastAI (`fastai`) -- 1.0.59
* Scikit-Learn (`sklearn`) -- 0.22
* TensorboardX -- If runtime profiling is not required, this can be removed.

## References 

<a id="1">[1]</a> 
R. Kiryo, G. Niu, M. Du Plessis, and M. Sugiyama. Positive-unlabeled learning with nonnegative risk estimator. In [NIPS, 2017](http://papers.nips.cc/paper/8072-co-teaching-robust-training-of-deep-neural-networks-with-extremely-noisy-labels).

<a id="2">[2]</a> 
Yu-Guan Hsieh, Gang Niu, and Masashi Sugiyama. Classification from positive, unlabeled and biased negative data. [arXiv:1810.00846, 2018](https://arxiv.org/abs/1810.00846).
