# plotter_utils/metrics/significance.py

import numpy as np
from .base import MetricBase, MetricResult

import logging
logger = logging.getLogger(__name__)

class SignificanceMetric(MetricBase):
    name = "significance"
    requires = ("sig", "bkg")
    #y_range = (0, 0.1)

    @staticmethod
    def safe_s_over_sqrtb_err(sig, bkg, sig_var=None, bkg_var=None):
        """
        Error propagation for S / sqrt(B).

        sigma^2 =
            (1/sqrt(B))^2 * sigma_S^2
          + (S / (2 B^(3/2)))^2 * sigma_B^2
        """
        sig = np.asarray(sig, dtype=float)
        bkg = np.asarray(bkg, dtype=float)

        y = np.divide(
            sig, np.sqrt(bkg),
            out=np.zeros_like(sig, dtype=float),
            where=bkg > 0
        )

        err2 = np.zeros_like(y, dtype=float)

        if sig_var is not None:
            err2 += np.divide(
                sig_var, bkg,
                out=np.zeros_like(sig_var, dtype=float),
                where=bkg > 0
            )

        if bkg_var is not None:
            err2 += np.divide(
                sig**2 * bkg_var, 4 * bkg**3,
                out=np.zeros_like(sig, dtype=float),
                where=bkg > 0
            )

        return np.sqrt(err2)

    def compute(self):
        sig_vals = self.to_1d(self.sig.values(flow=False))
        bkg_vals = self.to_1d(self.bkg.values_total(flow=False))

        sig_var = self.to_1d(self.sig.variances(flow=False))
        bkg_var = self.to_1d(self.bkg.hist_total.variances(flow=False))

        y = self.safe_divide(sig_vals, np.sqrt(bkg_vals))
        yerr = self.safe_s_over_sqrtb_err(
            sig_vals, bkg_vals,
            sig_var=sig_var,
            bkg_var=bkg_var,
        )

        x = self.sig.hist.axes[-1].centers

        logger.debug(f"S(x) {sig_vals[:10]} ...")
        logger.debug(f"B(x) {bkg_vals[:10]} ...")
        logger.debug(f"significance {y[:10]} ...")

        return MetricResult(
            x=x,
            y=y,
            yerr=yerr,
            label=r"$S/\sqrt{B}$"
        )