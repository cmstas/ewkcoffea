# plotter_utils/metrics/dataMC.py

import numpy as np
from .base import MetricBase, MetricResult

import logging
log = logging.getLogger(__name__)

class DataMCMetric(MetricBase):
    name = "dataMC"
    requires = ("data", "bkg")
    y_range = (0.5, 1.5)

    def compute(self):
        data_vals = self.to_1d(self.data.values(flow=False))
        bkg_vals  = self.to_1d(self.bkg.values_total(flow=False))

        data_var = self.to_1d(self.data.variances(flow=False))
        bkg_var  = self.to_1d(self.bkg.hist_total.variances(flow=False))

        ratio = self.safe_divide(data_vals, bkg_vals)
        ratio_err = self.safe_ratio_err(
            data_vals, bkg_vals,
            num_var=data_var,
            den_var=bkg_var,
        )

        x = self.data.hist.axes[-1].centers

        return MetricResult(
            x=x,
            y=ratio,
            yerr=ratio_err,
            label="Data / MC"
        )