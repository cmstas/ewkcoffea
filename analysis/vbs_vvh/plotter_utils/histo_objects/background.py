from plotter_utils.histo_objects.base import BaseHist
import ewkcoffea.modules.plotting_tools as plt_tools # type: ignore
import numpy as np # type: ignore


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
        h = plt_tools.merge_overflow(h)
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

    def get_yield_per_type(self):
        """
        Yield per background group (process_grp).
        """
        out = {}
        for grp in self.hist.axes["process_grp"]:
            h_grp = self.hist[{"process_grp": [grp]}]
            out[grp] = self._yield_from_hist(h_grp)
        return out

    def get_yield_per_process(self):
        """
        Yield per process, nested under background group.
        """
        out = {}

        # Loop over background groups (EWK, QCD, ...)
        for grp, proc_list in self.process_grouping.items():
            out[grp] = {}
            for proc in proc_list:
                # regroup THIS process alone
                h_proc = plt_tools.group(
                    self._raw_hist,
                    "process",
                    "process_grp",
                    {proc: [proc]},
                )
                # apply same pipeline steps
                h_proc = self._sum_years(h_proc)
                h_proc = h_proc[{"systematic": "nominal"}]
                h_proc = self._safe_merge_overflow(h_proc)

                out[grp][proc] = self._yield_from_hist(h_proc)

        return out

    def get_variance_per_type(self):
        out = {}
        for grp in self.hist.axes["process_grp"]:
            h_grp = self.hist[{"process_grp": [grp]}]
            out[grp] = self._variance_from_hist(h_grp)
        return out
    
    def get_uncertainty_per_type(self):
        out = {}
        for grp in self.hist.axes["process_grp"]:
            h_grp = self.hist[{"process_grp": [grp]}]
            out[grp] = np.sqrt(self._variance_from_hist(h_grp))
        return out