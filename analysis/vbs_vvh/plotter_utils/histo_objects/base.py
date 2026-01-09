import numpy as np
import copy
import ewkcoffea.modules.plotting_tools as plt_tools


class BaseHist:
    """
    Base class that handles transformations common to
    signal / background / data histograms.

    This class:
      - sums over years
      - groups raw processes into physics groups
      - exposes basic histogram information
      - provides a safe way to merge overflow bins
    """

    def __init__(
        self,
        hist,
        *,
        years,
        process_grouping,
    ):
        self._raw_hist = copy.deepcopy(hist) # freeze input
        self.years = years
        self.process_grouping = copy.deepcopy(process_grouping)

        # Prepare the final histogram immediately
        self._hist = self._prepare_hist()

    # ------------------------------------------------------------------
    # Core transformation pipeline
    # ------------------------------------------------------------------

    def _prepare_hist(self):
        """
        Apply transformations common to all histogram types.
        """
        h = self._raw_hist

        # 1) Sum over years
        h = self._sum_years(h)

        # 2) Group raw processes into physics processes
        h = self._group_processes(h)

        return h

    def _safe_merge_overflow(self, h, axis_slice=None):
        """
        Call merge_overflow safely:
          - optionally only on a subset of categories (axis_slice)
          - merge if dense axis has >= 2 bins
        """
        if axis_slice is not None:
            h = h[axis_slice]

        vals = h.values(flow=True)
        dense_bins = vals.shape[-1]

        if dense_bins >= 2:
            return plt_tools.merge_overflow(h)
        return h

    def _sum_years(self, h):
        """
        Collapse year axis into a single bin.
        """
        h = plt_tools.group(
            h,
            "year",
            "year_sum",
            {"all_years": self.years},
        )
        return h[{"year_sum": "all_years"}]

    def _group_processes(self, h):
        """
        Group raw MC samples into physics processes.
        """
        return plt_tools.group(
            h,
            "process",
            "process_grp",
            self.process_grouping,
        )
    
    def _yield_from_hist(self, h):
        return float(h.values(flow=True).sum())
        
    def _variance_from_hist(self, h):
        return float(h.variances(flow=True).sum())

    # ------------------------------------------------------------------
    # Public accessors (used by plotting & metrics)
    # ------------------------------------------------------------------

    @property
    def hist(self):
        return self._hist

    def values(self, flow=True):
        return self._hist.values(flow=flow)

    def variances(self, flow=True):
        return self._hist.variances(flow=flow)

    def total_yield(self):
        #return float(np.sum(self.values(flow=True)))
        return self._yield_from_hist(self._hist)
    
    def total_variance(self):
        return float(np.sum(self.variances(flow=True)))
    
    def uncertainty(self):
        return float(np.sqrt(self.total_variance()))
    
    def uncertainty_per_type(self):
        pass

    def bin_edges(self):
        return self._hist.axes[0].edges

    def bin_centers(self):
        return self._hist.axes[0].centers
    
    def get_yield_per_type(self):
        """
        Return yields grouped by 'type'.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_yield_per_process(self):
        """
        Return yields grouped by type -> process.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_variance_per_type(self):
        """Return yields grouped by 'type'."""
        raise NotImplementedError