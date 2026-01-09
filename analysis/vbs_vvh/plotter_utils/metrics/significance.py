# plotter_utils/metrics/significance.py

import numpy as np
from .base import MetricBase, MetricResult

import logging
log = logging.getLogger(__name__)

class SignificanceMetric(MetricBase):
    name = "significance"
    requires = ("sig", "bkg")
    y_range = (0, 0.1)

    def compute(self):
        sig_vals = self.to_1d(self.sig.values(flow=False))
        bkg_vals = self.to_1d(self.bkg.values_total(flow=False))

        log.debug(f"sig shape {sig_vals.shape}")
        log.debug(f"bkg shape {bkg_vals.shape}")
        log.debug(f"sig_vals {sig_vals[:10]}")
        log.debug(f"bkg_vals {bkg_vals[:10]}")

        y = self.safe_divide(sig_vals, np.sqrt(bkg_vals))
        log.debug(f"signif(y) shape {y.shape}")

        x = self.sig.hist.axes[-1].centers
        log.debug(f"bin(x) shape {x.shape}")

        log.debug(f"y {y[:10]}")
        log.debug(f"x {x[:10]}")

        return MetricResult(
            x=x,
            y=y,
            label=r"$S/\sqrt{B}$"
        )
