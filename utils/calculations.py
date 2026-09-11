import numpy as np
from scipy import signal, stats
import pandas as pd


def log_bin_edges(diams_nm: np.ndarray) -> np.ndarray:
    """Bin edges in log10(Dp), midway between centres and extrapolated at the ends."""
    log_d = np.log10(np.asarray(diams_nm, dtype=float))
    if len(log_d) < 2:
        return np.array([log_d[0] - 0.05, log_d[0] + 0.05])

    edges = np.empty(len(log_d) + 1)
    edges[1:-1] = (log_d[:-1] + log_d[1:]) / 2.0
    edges[0] = log_d[0] - (edges[1] - log_d[0])
    edges[-1] = log_d[-1] + (log_d[-1] - edges[-2])
    return edges


def dlogdp_per_bin(diams_nm: np.ndarray) -> np.ndarray:
    """Width of every size bin in log10(Dp).

    Integrate the distribution with these, not with one mean width: spliced or
    variable-resolution data has bins of genuinely different widths, and a single
    mean biases total number, mass and every sink that sums over the spectrum.
    """
    return np.diff(log_bin_edges(diams_nm))


def integrate_pnsd(pnsd_dndlogdp, dlogdp) -> np.ndarray:
    """Total number (cm-3) per row, from dN/dlogDp and per-bin (or uniform) widths."""
    return np.nansum(np.asarray(pnsd_dndlogdp, dtype=float) * np.asarray(dlogdp, dtype=float), axis=1)


def resolve_dlogdp(diams_nm: np.ndarray, typed_text=None) -> np.ndarray:
    """Per-bin widths, unless the user has typed a different uniform width.

    The panels show the mean width in an editable box.  Leaving it alone means
    "use the real bin widths"; changing it is an explicit uniform override.
    """
    widths = dlogdp_per_bin(diams_nm)
    if typed_text is None or typed_text == "":
        return widths
    try:
        typed = float(typed_text)
    except (TypeError, ValueError):
        return widths
    if typed <= 0 or np.isclose(typed, float(np.mean(widths)), rtol=5e-3):
        return widths
    return np.full(len(widths), typed)


def seconds_between(times) -> np.ndarray:
    """Seconds from each sample to the next, for real (possibly irregular) data.

    Passing a single number treats the series as evenly spaced by that many
    seconds, which is what the fixed-grid diurnal composites want.
    """
    if np.isscalar(times):
        return np.asarray([float(times)])
    idx = pd.DatetimeIndex(times)
    return np.diff(idx.view("int64")) / 1e9


def get_coagulation_coef(d_nm: np.ndarray, T: float = 293.15):
    """Calculates the Coagulation Coefficient matrix K (m3/s)."""
    d = d_nm * 1e-9                                                          
    dij = np.add.outer(d, d)                                                 
    
    Kn = (2 * 65e-9) / d                                                     
    mu = 1.7e-5                                                              
    C = 1 + Kn * (1.257 + 0.4 * np.exp(-(1.10 / Kn)))                        
    D = (1.3806e-23 * T * C) / (3 * np.pi * mu * d)                          
    m = ((4/3) * np.pi * (d/2)**3) * 1.83e3                                  
    c = np.sqrt((8 * 1.3806e-23 * T) / (np.pi * m))                          
    
    yi = (8 * D) / (np.pi * c)                                               
    omega = (((d + yi)**3 - (d**2 + yi**2)**(3/2)) / (3 * d * yi)) - d       
    
    Dij = np.add.outer(D, D)                                                 
    cij = np.sqrt(np.add.outer(c**2, c**2))                                  
    oij = np.sqrt(np.add.outer(omega**2, omega**2))                          
    
    Kc = 4 * np.pi * dij * Dij                                               
    K = Kc / ((dij / (dij + oij)) + (4 * Dij / (cij * dij)))                 
    
    return K                                                                 

def calc_coagulation_sink(diams_nm: np.ndarray, pnsd_dndlogdp: np.ndarray, dlogdp: float, T=293.15):
    """Calculates Coagulation Sink (s-1) for each size bin across the entire time series."""
    K = get_coagulation_coef(diams_nm, T)                                    # Get 2D coagulation matrix
    # Traditional models underestimate the sink by assuming a single 'representative' diameter, 
    # which artificially minimises the collision coefficient (Beta).
    # Here, we calculate discrete collisions across the entire size distribution.
    # Applying an upper-triangular mask (np.triu) captures the NET population loss 
    # (scavenging by equal or larger particles) without double-counting.
    K_upper = np.tril(K)                                                     # make sure particles only coagulate with particles of a larger diameter :)
    N_m3 = (pnsd_dndlogdp * dlogdp) * 1e6                                    # Convert to actual N (m-3)
    CoagS_matrix = np.dot(N_m3, K_upper)                                     # Vectorised sum against larger particles only
    return CoagS_matrix                                                      # Returns shape (time, bins)

def calc_condensation_sink(diams_nm: np.ndarray, pnsd_dndlogdp: np.ndarray, dlogdp: float, T=293.15, P=101.325):
    """Calculates Condensation Sink (s-1) using Fuchs-Sutugin."""
    d = diams_nm * 1e-9                                                      
    Kn = (2 * 65e-9) / d                                                     
    betaM = (Kn + 1) / (1 + 1.677 * Kn + 1.333 * Kn**2)                      
    
    # Fuller diffusion volumes for the air/H2SO4 pair, and the reduced molar mass.
    Mair, Msulp = 28.965, 98.079
    M_AB = 2.0 / (1.0 / Mair + 1.0 / Msulp)
    dair, dsulp = 19.7, 22.9 + 6.11*4 + 2.31*2

    # Fuller (1966): D[cm2/s] = 0.00143 T^1.75 / (P[bar] sqrt(M_AB) (sv_A^1/3 + sv_B^1/3)^2).
    # T is raised to 1.75, not (0.00143 T); P is in bar, not kPa.
    P_bar = P / 100.0
    D_cm2 = (0.00143 * T**1.75) / (P_bar * np.sqrt(M_AB) * (dair**(1/3) + dsulp**(1/3))**2)
    D = D_cm2 * 1e-4                                                         # cm2/s -> m2/s

    N_m3 = (pnsd_dndlogdp * dlogdp) * 1e6
    cs_series = 2 * np.pi * D * np.nansum(N_m3 * betaM * d, axis=1)
    return cs_series

def calc_formation_rate(diams_nm: np.ndarray, pnsd_dndlogdp: np.ndarray, dlogdp,
                        gr_nm_hr: float, j_min_nm: float, j_max_nm: float, coags_matrix: np.ndarray,
                        times):
    """Calculates Formation Rate (J) using the exact bounds and Coagulation Sink matrix.

    ``times`` is the timestamp index of the rows, or a single number of seconds
    for evenly spaced data.  dN/dt has to use the real sampling interval: taking
    it as one hour makes J wrong by the ratio for 15 min or 10 min data.
    """
    j_mask = (diams_nm >= j_min_nm) & (diams_nm <= j_max_nm)                 # Isolate the J boundary bins

    widths = np.asarray(dlogdp, dtype=float)
    widths = widths[j_mask] if widths.ndim else widths                       # per-bin widths, if given
    N_j = pnsd_dndlogdp[:, j_mask] * widths                                  # Target bins dN
    Bin_total = np.nansum(N_j, axis=1)                                       # Total N in target range per row

    dt = seconds_between(times)
    dN_dt = np.zeros_like(Bin_total)
    with np.errstate(divide="ignore", invalid="ignore"):
        dN_dt[1:] = np.diff(Bin_total) / dt
    dN_dt[~np.isfinite(dN_dt)] = 0.0                                         # repeated timestamps, if any
    
    # Calculate row-by-row weighted mean CoagS for the target bins
    weights = np.zeros_like(N_j)
    valid_rows = Bin_total > 0
    weights[valid_rows] = N_j[valid_rows] / Bin_total[valid_rows, None]
    
    mean_coags = np.sum(weights * coags_matrix[:, j_mask], axis=1)           # Apply weights to CoagS
    coag_term = mean_coags * Bin_total                                       
    
    gr_term = (gr_nm_hr / (3600.0 * (j_max_nm - j_min_nm))) * Bin_total      
    
    j_total = dN_dt + coag_term + gr_term                                    
    return j_total, dN_dt, gr_term, coag_term

# ----------------------------------------------------------------------------------------------- #
# Use this to estimate J1.5 from Jx :)
# ----------------------------------------------------------------------------------------------- #
def calculate_m(coags_d1: float, coags_dx: float, d1: float, dx: float) -> float:
    """Calculates the power-law exponent 'm' for the coagulation sink."""
    return np.log(coags_dx / coags_d1) / np.log(dx / d1)

def calculate_j1_5(d1: float, Jx: float, dx: float, coags_d1_5: float, gr: float, m: float) -> float:
    """
    Calculates J1.5 from Jx using the Kerminen-Kulmala survival equation.
    IMPORTANT: Ensure coags_d1_5 and gr are in compatible time units!
    (e.g., if GR is nm/h, CoagS must be converted to h^-1 before passing).
    """
    # Calculate the correction factor (xi or gamma)
    xi = (1 / (m + 1)) * ((dx / d1)**(m + 1) - 1)
    
    # Calculate J1.5 using the exponential relationship
    # Note: Your R code used dx in the exponent. Standard Lehtinen 2007 uses d1.
    J1_5 = Jx * np.exp(xi * dx * coags_d1_5 / gr) 
    
    return J1_5

def fit_modes_to_pnsd(subset_pnsd: np.ndarray, active_diams: np.ndarray, limit_jump: bool = True, max_jump: float = 15.0, overrides: dict = None):
    """Extracts the dominant mode, with absolute user overrides for shoulders/shelves."""
    if overrides is None: overrides = {}                                             
    
    mode_diams = []                                                                  
    valid_indices = []                                                               
    fit_snapshots = []                                                               
    last_peak_dp = None                                                              
    
    for i, row in enumerate(subset_pnsd):                                            
        peaks, _ = signal.find_peaks(row, prominence=np.max(row)*0.05)               
        peaks = list(peaks)                                                          
        
        if len(row) > 1 and row[0] > row[1]:                                         
            peaks.append(0)                                                          
        if len(row) > 1 and row[-1] > row[-2]:                                       
            peaks.append(len(row) - 1)                                               
            
        best_peak = None                                                             
        
        if len(peaks) > 0 or i in overrides:                                                           
            if i in overrides:                                                       
                target_dp = overrides[i]                                             
                # ABSOLUTE OVERRIDE: Ignore peaks, find the exact bin closest to the click!
                best_peak = np.argmin(np.abs(active_diams - target_dp))                
            elif last_peak_dp is not None and limit_jump:                            
                valid_peaks = [p for p in peaks if np.abs(active_diams[p] - last_peak_dp) <= max_jump]
                if valid_peaks: 
                    best_peak = valid_peaks[np.argmax(row[valid_peaks])]             
            elif len(peaks) > 0:                                                                    
                best_peak = peaks[np.argmax(row[peaks])]                             
        
        if best_peak is not None:                                                    
            peak_dp = active_diams[best_peak]                                        
            mode_diams.append(peak_dp)                                               
            valid_indices.append(i)                                                  
            fit_snapshots.append((active_diams, row, peak_dp))                       
            last_peak_dp = peak_dp                                                   
            
    return valid_indices, mode_diams, fit_snapshots

def calc_growth_rate(time_hours: np.ndarray, mode_diams: np.ndarray):
    """Calculates GR (nm/hr) using simple linear regression."""
    res = stats.linregress(time_hours, mode_diams)                                   # Perform linear fit
    return res.slope, res.intercept                                                  # Return slope and intercept