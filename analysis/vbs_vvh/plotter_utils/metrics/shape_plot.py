# plotter_utils/metrics/shape_plot.py

import numpy as np
from .base import MetricBase, MetricResult


class ShapePlotMetric(MetricBase):
    name = "shape_plot"
    requires = ()
    y_range = None

    @staticmethod
    def _normalize(vals):
        total = vals.sum()
        if total <= 0:
            return np.zeros_like(vals, dtype=float)
        return vals / total

    def compute(self):
        curves = {}

        if self.sig is not None:
            vals = self.to_1d(self.sig.values(flow=False))
            curves["Signal"] = self._normalize(vals)

        if self.bkg is not None:
            vals = self.to_1d(self.bkg.values_total(flow=False))
            curves["Background"] = self._normalize(vals)

        if self.data is not None:
            vals = self.to_1d(self.data.values(flow=False))
            curves["Data"] = self._normalize(vals)

        if not curves:
            return MetricResult(x=[], y={}, label="Normalized shape")

        hist = self.sig or self.bkg or self.data
        x = hist.hist.axes[-1].centers

        return MetricResult(
            x=x,
            y=curves,
            label="Normalized shape",
        )
