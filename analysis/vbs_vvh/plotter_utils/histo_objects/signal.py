from plotter_utils.histo_objects.base import BaseHist


class SignalHist(BaseHist):
    """
    Signal histogram.

    Differences from BaseHist:
      - select ONLY signal process
      - select coupling-dependent systematic
    """

    def __init__(
        self,
        hist,
        *,
        years,
        process_grouping,
        coupling,
    ):
        self.coupling = coupling
        super().__init__(
            hist,
            years=years,
            process_grouping=process_grouping,
        )

    def _prepare_hist(self):
        h = super()._prepare_hist()

        # Select signal + coupling systematic
        h = h[
            {
                "process_grp": ["Signal"],
                "systematic": self.coupling,
            }
        ]

        # Safe merge overflow for signal
        h = self._safe_merge_overflow(h)
        return h

    def cumulative(self):
        return self.values(flow=False).cumsum()
    
    def get_yield_per_type(self):
        # No type layer → alias to per_process
        return self.get_yield_per_process()['Signal']

    def get_yield_per_process(self):
        out = {"Signal": {}}
        for proc in self._raw_hist.axes["process"]:
            h_proc = self._raw_hist[{"process": [proc]}]
            out["Signal"][proc] = self._yield_from_hist(h_proc)
        return out

    def get_variance_per_type(self):
        return {"Signal":self.total_variance()}