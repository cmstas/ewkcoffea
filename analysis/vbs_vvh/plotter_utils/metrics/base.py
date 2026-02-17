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
    
    #errorbars
    @staticmethod
    def poisson_err(vals):
        """
        Poisson statistical uncertainty.
        Negative or NaN inputs give zero error.
        """
        vals = np.asarray(vals, dtype=float)
        return np.sqrt(np.clip(vals, 0.0, None))

    @staticmethod
    def safe_ratio_err(num, den, num_var=None, den_var=None):
        """
        Error propagation for ratio R = num / den.

        If only den_var is provided:
            sigma_R = R * sqrt(den_var / den^2)

        If both num_var and den_var are provided:
            sigma_R = R * sqrt( num_var/num^2 + den_var/den^2 )

        Returns zero where den <= 0.
        """
        num = np.asarray(num, dtype=float)
        den = np.asarray(den, dtype=float)

        R = np.divide(
            num, den,
            out=np.zeros_like(num, dtype=float),
            where=den > 0
        )

        err2 = np.zeros_like(R, dtype=float)

        if num_var is not None:
            num_var = np.asarray(num_var, dtype=float)
            err2 += np.divide(
                num_var, num**2,
                out=np.zeros_like(num_var, dtype=float),
                where=num > 0
            )

        if den_var is not None:
            den_var = np.asarray(den_var, dtype=float)
            err2 += np.divide(
                den_var, den**2,
                out=np.zeros_like(den_var, dtype=float),
                where=den > 0
            )

        return R * np.sqrt(err2)

