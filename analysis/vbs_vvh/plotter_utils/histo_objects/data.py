from plotter_utils.histo_objects.base import BaseHist


class DataHist(BaseHist):
    """
    Data histogram.

    Differences from BaseHist:
      - select Data process
      - nominal only
    """

    def _prepare_hist(self):
        h = super()._prepare_hist()
        return h[
            {
                "process_grp": ["Data"],
                "systematic": "nominal",
            }
        ]

    def get_yield_per_type(self):
        return self.get_yield_per_process()['Data']

    def get_yield_per_process(self):
        out = {"Data": {}}

        for proc in self._raw_hist.axes["process"]:
            h_proc = self._raw_hist[{"process": [proc]}]
            out["Data"][proc] = self._yield_from_hist(h_proc)

        return out
    
    def get_variance_per_type(self):
        return {"Data":self.total_variance()}