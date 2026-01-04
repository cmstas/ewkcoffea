from plotter_utils.histo_objects.base import BaseHist
from plotter_utils.helpers.funcs import combine
import ewkcoffea.modules.plotting_tools as plt_tools


class BackgroundHist(BaseHist):
    """
    Background histogram.

    Differences from BaseHist:
      - always nominal systematic
      - keeps individual background process_grp for stacked plotting
      - provides total background histogram
    """

    def __init__(
        self,
        hist,
        *,
        years,
        process_grouping,
        background_groups,
    ):
        self.background_groups = background_groups
        super().__init__(
            hist,
            years=years,
            process_grouping=process_grouping,
        )

        # After preparing the histogram, generate the "total" histogram
        self._hist_total = self._make_total_background(self._hist)

    def _prepare_hist(self):
        import copy
        h = super()._prepare_hist()

        # Nominal only
        h = h[{"systematic": "nominal"}]

        # # Safe merge overflow **per process_grp**
        # hist_slices = {}
        # for proc in h.axes["process_grp"]:
        #     print(f'doing {proc}')
        #     hist_slices[proc] = self._safe_merge_overflow(h, {"process_grp": [proc]})
        # # Combine back
        # print(h)
        h = plt_tools.merge_overflow(h)
        #h = combine(hist_slices)
        return h

    def _make_total_background(self, h):
        """
        Collapse all background processes into one for total yield calculations.
        """
        h_total = plt_tools.group(
            h,
            "process_grp",
            "sample_category",
            {"Background": list(self.background_groups)},
        )
        h_total = self._safe_merge_overflow(h_total)
        return h_total

    @property
    def hist_total(self):
        """
        Collapsed total background histogram.
        """
        return self._hist_total

    def values_total(self, flow=True):
        return self._hist_total.values(flow=flow)

    def cumulative_total(self):
        return self.values_total(flow=False).cumsum()

    def cumulative(self):
        """
        Cumulative per-process background yield (stacked)
        """
        return self.values(flow=False).cumsum()
