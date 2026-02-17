# plotter_utils/metrics/pass_rate.py

import numpy as np
from .base import MetricBase, MetricResult


class EfficiencyMetric(MetricBase):
    name = "efficiency"
    requires = ()
    y_range = (0, 1.05)

    def compute(self):
        curves = {}

        if self.sig is not None:
            passed = self.to_1d(self.sig.left_cumulative())
            total = passed[-1] if passed.size else 0.0
            curves["Signal"] = passed / total if total > 0 else passed * 0.0

        if self.bkg is not None:
            passed = self.to_1d(self.bkg.left_cumulative(total=True))
            total = passed[-1] if passed.size else 0.0
            curves["Background"] = passed / total if total > 0 else passed * 0.0

        if self.data is not None:
            passed = self.to_1d(self.data.left_cumulative())
            total = passed[-1] if passed.size else 0.0
            curves["Data"] = passed / total if total > 0 else passed * 0.0

        if not curves:
            return MetricResult(x=[], y={}, label="Efficiency")

        hist = self.sig or self.bkg or self.data
        x = hist.hist.axes[-1].centers

        return MetricResult(
            x=x,
            y=curves,
            label="Efficiency",
        )
