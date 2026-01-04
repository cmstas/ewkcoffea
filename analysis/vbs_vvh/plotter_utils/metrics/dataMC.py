# plotter_utils/metrics/dataMC.py

import numpy as np
from .base import MetricBase, MetricResult

import logging
log = logging.getLogger(__name__)

class DataMCMetric(MetricBase):
    name = "dataMC"
    requires = ("data", "bkg")
    y_range = (0.3, 3)

    def compute(self):
        data_vals = self.to_1d(self.data.values(flow=False))
        bkg_vals = self.to_1d(self.bkg.values_total(flow=False))

        ratio = self.safe_divide(data_vals, bkg_vals)

        x = self.data.hist.axes[-1].centers

        return MetricResult(
            x=x,
            y=ratio,
            label="Data / MC"
        )
