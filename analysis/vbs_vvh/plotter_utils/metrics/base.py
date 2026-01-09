# plotter_utils/metrics/base.py

import numpy as np
from abc import ABC, abstractmethod

import logging
log = logging.getLogger(__name__)


class MetricResult:
    """
    Container passed from metrics -> plotting.
    """
    def __init__(self, x, y, yerr=None, label=None):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.label = label


class MetricBase(ABC):
    name = None
    requires = ()   # ("sig", "bkg", "data")
    y_range = None   # (ymin, ymax)

    def __init__(self, *, sig=None, bkg=None, data=None):
        self.sig = sig
        self.bkg = bkg
        self.data = data

    def available(self):
        missing = [k for k in self.requires if getattr(self, k) is None]
        return len(missing) == 0, missing

    @staticmethod
    def safe_divide(num, den):
        return np.divide(
            num, den,
            out=np.full_like(num, np.nan, dtype=float),
            where=den > 0
        )

    @abstractmethod
    def compute(self):
        """
        Return MetricResult
        """
        pass
    
    @staticmethod
    def to_1d(arr):
        """
        Collapse all non-dense axes.
        Expects dense axis to be the last one.
        """
        arr = np.asarray(arr)

        # Sum over all axes except the last (dense) one
        if arr.ndim > 1:
            axes = tuple(range(arr.ndim - 1))
            arr = arr.sum(axis=axes)

        return arr