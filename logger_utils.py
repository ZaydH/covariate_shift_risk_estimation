# -*- utf-8 -*-
r"""
    logger_utils.py
    ~~~~~~~~~~~~~~~

    Provides utilities to simplify and standardize logging in particular for training for training
    with \p torch.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""
try:
    # noinspection PyUnresolvedReferences
    import matplotlib
    # noinspection PyUnresolvedReferences
    from matplotlib import use
    use('Agg')
except ImportError:
    # raise ImportError("Unable to import matplotlib")
    pass

from datetime import datetime
import logging
import re
import sys
from typing import Optional

import torch

from pubn import BASE_DIR
from pubn.custom_types import ListOrInt
from pubn.logger import FORMAT_STR, create_stdout_handler


def setup_logger(log_level: int = logging.DEBUG, job_id: Optional[ListOrInt] = None) -> None:
    r"""
    Logger Configurator

    Configures the test logger.

    :param log_level: Level to log
    :param job_id: Identification number for the job
    """
    date_format = '%m/%d/%Y %I:%M:%S %p'  # Example Time Format - 12/12/2010 11:46:36 AM

    job_str, time_str = "", str(datetime.now()).replace(" ", "-")
    if job_id is not None:
        if isinstance(job_id, int): job_id = [job_id]
        job_str = "_j=%s_" % "-".join("%05d" % x for x in job_id)
    filename = "log%s_%s.log" % (job_str, time_str)

    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = log_dir / filename

    logging.basicConfig(filename=filename, level=log_level, format=FORMAT_STR, datefmt=date_format)

    # Print to stdout on root logger if no handlers exist
    if not hasattr(setup_logger, "_HAS_ROOT"):
        create_stdout_handler(log_level, format_str=FORMAT_STR)
        setup_logger._HAS_ROOT = True

    # Matplotlib clutters the logger so change its log level
    if "matplotlib" in sys.modules:
        # noinspection PyProtectedMember
        matplotlib._log.setLevel(logging.WARNING)  # pylint: disable=protected-access
    # Disable logging in PyTorch ignite
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    logging.info("******************* New Run Beginning *****************")
    # noinspection PyUnresolvedReferences
    logging.debug("Torch Version: %s", torch.__version__)
    logging.debug("Torch CUDA: %s", "ENABLED" if torch.cuda.is_available() else "Disabled")
    # noinspection PyUnresolvedReferences
    logging.debug("Torch cuDNN Enabled: %s", "YES" if torch.backends.cudnn.is_available() else "NO")
    logging.info(" ".join(sys.argv))
    logging.debug("Torch Random Seed: %d", torch.initial_seed())
    if "numpy" in sys.modules:
        import numpy as np
        state_str = re.sub(r"\s+", " ", str(np.random.get_state()))
        logging.debug("NumPy Random Seed: %s", state_str)
    #  Prints a seed way too long for normal use
    # if "random" in sys.modules:
    #     import random
    #     logging.debug("Random (package) Seed: %s", random.getstate())
