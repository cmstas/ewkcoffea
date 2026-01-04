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
