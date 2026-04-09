import numpy as np

def condensational_sink(
    pnsd: np.ndarray,   # shape (n_times, n_bins), dN/dlogDp in cm-3
    d_nm: np.ndarray,   # bin centre diameters in nm
    dlogdp: float,
    T: float = 293.0,
    P: float = 101.325,
) -> np.ndarray:
    """
    Returns condensational sink (CS) in s-1 for each time step.
    """
    d = d_nm * 1e-9                          # nm → m
    Kn = (2 * 65e-9) / d
    betaM = (Kn + 1) / (1 + 1.677 * Kn + 1.333 * Kn**2)

    Mair = 28.965
    dair, dsulp = 19.7, 22.9 + 6.11*4 + 2.31*2
    D = ((0.00143 * T)**1.75) / (P * np.sqrt(Mair) * (dair**(1/3) + dsulp**(1/3))**2)

    N = (pnsd / dlogdp) * 1e6               # dN/dlogDp → absolute N in m-3
    CS = 2 * np.pi * D * (N * betaM * d).sum(axis=1)
    return CS
