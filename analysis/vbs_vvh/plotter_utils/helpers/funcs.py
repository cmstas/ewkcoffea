import numpy as np

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
