import copy
import hist



def combine(hist_slices: dict) -> hist.Hist:
    #working on it not used for now
    """
    Combine multiple single-category Hist objects into a single Hist
    with an axis containing all categories. Works with Weight() storage.
    """
    if not hist_slices:
        raise ValueError("No histograms to combine.")

    first_hist = next(iter(hist_slices.values()))
    cat_axis_name = first_hist.axes[0].name  # assume first axis is categorical
    dense_axes = [ax for ax in first_hist.axes if ax.name != cat_axis_name]

    categories = list(hist_slices.keys())
    hnew = hist.Hist(
        hist.axis.StrCategory(categories, name=cat_axis_name),
        *dense_axes,
        storage=first_hist.storage_type()
    )

    for cat in categories:
        # Use dict-style slicing to assign values safely
        hnew[{cat_axis_name: cat}] = hist_slices[cat]

    return hnew