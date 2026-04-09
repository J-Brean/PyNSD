import numpy as np
from scipy import signal, stats
import pandas as pd

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
    
    Mair, dair, dsulp = 28.965, 19.7, 22.9 + 6.11*4 + 2.31*2                 
    D = ((0.00143 * T)**1.75) / (P * np.sqrt(Mair) * (dair**(1/3) + dsulp**(1/3))**2) 
    
    N_m3 = (pnsd_dndlogdp * dlogdp) * 1e6                                    
    cs_series = 2 * np.pi * D * np.sum(N_m3 * betaM * d, axis=1)             
    return cs_series                                                         

def calc_formation_rate(diams_nm: np.ndarray, pnsd_dndlogdp: np.ndarray, dlogdp: float, 
                        gr_nm_hr: float, j_min_nm: float, j_max_nm: float, coags_matrix: np.ndarray):
    """Calculates Formation Rate (J) using the exact bounds and Coagulation Sink matrix."""
    j_mask = (diams_nm >= j_min_nm) & (diams_nm <= j_max_nm)                 # Isolate the J boundary bins
    
    N_j = pnsd_dndlogdp[:, j_mask] * dlogdp                                  # Target bins dN
    Bin_total = np.sum(N_j, axis=1)                                          # Total N in target range per row
    
    dN_dt = np.zeros_like(Bin_total)                                         
    dN_dt[1:] = np.diff(Bin_total) / 3600.0                                  
    
    # Calculate row-by-row weighted mean CoagS for the target bins
    weights = np.zeros_like(N_j)
    valid_rows = Bin_total > 0
    weights[valid_rows] = N_j[valid_rows] / Bin_total[valid_rows, None]
    
    mean_coags = np.sum(weights * coags_matrix[:, j_mask], axis=1)           # Apply weights to CoagS
    coag_term = mean_coags * Bin_total                                       
    
    gr_term = (gr_nm_hr / (3600.0 * (j_max_nm - j_min_nm))) * Bin_total      
    
    j_total = dN_dt + coag_term + gr_term                                    
    return j_total, dN_dt, gr_term, coag_term

from scipy import signal, stats
import numpy as np

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


def resample_wind_data(wind_df: pd.DataFrame, target_index: pd.DatetimeIndex, ws_col: str, wd_col: str):
    """Resamples high-resolution WS/WD data to match PNSD timestamps using true vector averaging."""
    wd_rad = np.radians(wind_df[wd_col])                                     # Convert degrees to radians
    
    wind_df['u'] = -wind_df[ws_col] * np.sin(wd_rad)                         # East-West (u) vector component
    wind_df['v'] = -wind_df[ws_col] * np.cos(wd_rad)                         # North-South (v) vector component
    
    freq = pd.infer_freq(target_index) or '1H'                               # Infer PNSD resolution (usually 1H)
    resampled = wind_df[['u', 'v']].resample(freq).mean()                    # Average the raw vectors
    
    res_ws = np.sqrt(resampled['u']**2 + resampled['v']**2)                  # Reconstruct true wind speed
    res_wd = (np.degrees(np.arctan2(-resampled['u'], -resampled['v'])) + 360) % 360 # Reconstruct true wind direction
    
    final_df = pd.DataFrame({'WS': res_ws, 'WD': res_wd}, index=resampled.index) # Create clean dataframe
    return final_df.reindex(target_index, method='nearest', tolerance=pd.Timedelta(freq)) # Snap exactly to PNSD timestamps

def assign_wind_sectors(df: pd.DataFrame, sectors: list):
    """Assigns data to wind sectors, safely handling the 360-0 degree wrap-around."""
    df['Sector'] = 'Unclassified'                                            # Default state
    for sec in sectors:                                                      # Loop through user-defined limits
        name, s_min, s_max = sec['name'], float(sec['min']), float(sec['max']) 
        if s_min > s_max:                                                    # Handle North wrap-around (e.g., 330 to 30)
            mask = (df['WD'] >= s_min) | (df['WD'] <= s_max)                 # Use OR operator
        else:                                                                # Standard slice
            mask = (df['WD'] >= s_min) & (df['WD'] <= s_max)                 # Use AND operator
        df.loc[mask, 'Sector'] = name                                        # Assign name to matching rows
    return df                                                                # Return updated dataframe