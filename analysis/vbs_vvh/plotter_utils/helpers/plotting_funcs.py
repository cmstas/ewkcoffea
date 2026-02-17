import numpy as np
import re

def snap_to_decade(factor):
    """
    Snap factor to a * 10^n with a in [1..9]. Used in rounding off for scaling sig to bkg
    """
    if factor <= 0:
        return 1.0

    n = int(np.floor(np.log10(factor)))
    a = factor / (10 ** n)

    # Clamp a to [1, 9] and round to nearest integer
    a = int(np.clip(np.round(a), 1, 9))

    return a * (10 ** n)


def plt_scientific_notation(ax,limit=4):
    """ set scientific notation if y-axis size >= 10e(limit) """
    import matplotlib.ticker as mticker

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((limit, limit))  # switch at 10^4

    ax.yaxis.set_major_formatter(formatter)
    return ax

def format_variable_label(hist):
    """ fixing this
            raise s.error('bad escape %s' % this, len(this))
            re.error: bad escape \D at position 1
    """
    axis = hist.axes[-1]
    label = axis.label or axis.name
    s = label

    rules = [
        # Delta variables (with or without underscore)
        (r"(?:^|_)dR\b",    "$\\Delta R$"),
        (r"(?:^|_)dphi\b",  "$\\Delta \\phi$"),
        (r"(?:^|_)deta\b",  "$\\Delta \\eta$"),

        # Single variables
        (r"\bphi\b", "$\\phi$"),
        (r"\beta\b", "$\\eta$"),
    ]

    for pattern, repl in rules:
        s = re.sub(pattern, repl, s)

    # Optional cosmetic cleanup
    s = s.replace("_", " ")

    return s