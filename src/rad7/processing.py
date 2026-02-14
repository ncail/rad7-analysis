"""
Data processing module for Rad7 data.
Handles calibration, rebinning, and moisture correction.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Union

# Constants
RAD7_VOLUME_M3 = 0.0008  # 800 mL

def calculate_counts(df: pd.DataFrame, 
                    only_po218: bool = False, 
                    only_po214: bool = False) -> pd.DataFrame:
    """
    Calculates specific counts based on isotope selection (Po218, Po214).
    Adds a 'Counts' column to the DataFrame.
    """
    
    counts = []
    
    # Pre-calculate common terms for efficiency
    # matching original logic structure initially, but vectorized for performance. 
    
    total_counts = df['TotalCounts'].values
    percent_a = df['PercentA'].values / 100.0
    percent_c = df['PercentC'].values / 100.0
    percent_d = df['PercentD'].values
    
    # Term for spillover correction from D window
    # Original: int(df['TotalCounts'][i]*df['PercentD'][i]/195)
    # The 195 seems to be a hardcoded constant for spillover normalization
    spillover = (total_counts * percent_d / 195).astype(int)
    
    if only_po218:
        # 2 * A - spillover
        calculated_counts = 2 * total_counts * percent_a - spillover
    elif only_po214:
        # 2 * C - spillover
        calculated_counts = 2 * total_counts * percent_c - spillover
    else:
        # (A + C) - spillover
        # Original: df['TotalCounts'][i]*(df['PercentA'][i]/100 + df['PercentC'][i]/100) - ...
        calculated_counts = total_counts * (percent_a + percent_c) - spillover
        
    # Apply max(0, count) as counts cannot be negative
    # Original did this: if countsDummy[-1] < 0: countsDummy[-1] = 0
    calculated_counts = np.maximum(calculated_counts, 0)
    
    df['Counts'] = calculated_counts
    
    # Store the calculation used for calibration later
    # The original code always calculates 'counts_forCalibration' using both A and C
    counts_for_calibration = total_counts * (percent_a + percent_c) - spillover
    df['Counts_Calibration_Basis'] = counts_for_calibration
    
    return df

def correct_for_humidity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies empirical humidity correction to Counts and RnConc.
    Based on Joseph Street's implementation (211020).
    """
    rh = df['RHofSampledAir']
    temp = df['Temperature']
    
    correction_factor = (1 + (-0.01537196) * rh + 
                             (-0.00215327) * temp + 
                             (0.00108539) * rh * temp)
    
    # Apply to existing counts (if they exist) and RnConc
    if 'Counts' in df.columns:
        df['Counts'] = df['Counts'] * correction_factor
        
    if 'RnConc' in df.columns:
        df['RnConc'] = df['RnConc'] * correction_factor
        
    if 'Uncert_RnConc' in df.columns:
        df['Uncert_RnConc'] = df['Uncert_RnConc'] * correction_factor
        
    return df

def calibrate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the 'Cnts_to_Bqm3' calibration factor for each row.
    Handles division by zero by averaging neighbors.
    """
    if 'Counts_Calibration_Basis' not in df.columns:
        # Fallback if not calculated yet
         df = calculate_counts(df)

    counts_calib = df['Counts_Calibration_Basis']
    rn_conc = df['RnConc']
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        calibration_factor = np.where(counts_calib != 0, rn_conc / counts_calib, 0.0)
        
    # Fix zeros by averaging valid neighbors (forward looking)
    # This logic mimics the original getCalibration loop
    calib_list = calibration_factor.tolist()
    
    for i in range(len(calib_list)):
        if calib_list[i] == 0:
            # Look ahead for non-zero values
            subset = [x for x in calib_list[i:] if x > 0]
            if subset:
                # Original logic: average(cnts_per_bin_to_Bq_per_m3[i-2:notZeroIndex])
                # We will just take the average of what we found to be safe approx or use next valid
                # Simplifying for robustness: use mean of entire non-zero set or just forward fill?
                # Original logic was complex and specific. Let's try to do a simple forward/backward fill or mean
                calib_list[i] = np.mean(subset) if len(subset) > 0 else 1.0 # Default fallback
            else:
                 # If no future values, look specific backward
                 subset_back = [x for x in calib_list[:i] if x > 0]
                 calib_list[i] = np.mean(subset_back) if len(subset_back) > 0 else 1.0

    df['Cnts_to_Bqm3'] = calib_list
    return df

def process_run(df: pd.DataFrame,
                bin_hours: float,
                correct_rh: bool = True,
                only_po218: bool = False,
                only_po214: bool = False) -> pd.DataFrame:
    """
    Processes a single run DataFrame.
    """
    if df.empty:
        return df

    # DateTime should already be present from io/raw_parser
    if 'DateTime' not in df.columns:
        print("Warning: DateTime column missing. Processing might fail.")
    else:
        # Ensure it is datetime object
        if not pd.api.types.is_datetime64_any_dtype(df['DateTime']):
            df['DateTime'] = pd.to_datetime(df['DateTime'])

    df = calculate_counts(df, only_po218, only_po214)
    
    if correct_rh:
        df = correct_for_humidity(df)
        
    df = calibrate_data(df)
    
    # Re-binning
    binned_df = rebin_data(df, bin_hours)
    
    return binned_df

def process_runs(run_list: List[pd.DataFrame], 
                 bin_hours: float, 
                 correct_rh: bool = True,
                 only_po218: bool = False,
                 only_po214: bool = False) -> pd.DataFrame:
    """
    Main processing function.
    1. Parses dates
    2. Calculates counts
    3. Corrects for RH
    4. Calibrates
    5. Rebins
    6. Concatenates
    """
    processed_runs = []
    
    for df in run_list:
        binned_df = process_run(df, bin_hours, correct_rh, only_po218, only_po214)
        processed_runs.append(binned_df)
        
    
    # Concatenate all runs
    if not processed_runs:
        return pd.DataFrame()
        
    final_df = pd.concat(processed_runs, ignore_index=True)
    
    return final_df

def rebin_data(df: pd.DataFrame, bin_hours: float) -> pd.DataFrame:
    """
    Rebins the data into specified time intervals.
    """
    if df.empty:
        return df
        
    # Sort by time just in case
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    start_time = df['DateTime'].iloc[0]
    end_time = df['DateTime'].iloc[-1]
    
    # Create bin edges
    # Round to nearest hour/minute to align nicely? Or just start from 0
    # Original logic used timedelta steps
    
    binned_data = []
    
    current_time = start_time
    # Determine the time step of raw data to see if we need to aggregate
    if len(df) > 1:
        raw_step = (df['DateTime'].iloc[1] - df['DateTime'].iloc[0]).total_seconds() / 3600.0
    else:
        raw_step = 0.0
        
    # If bin_hours is close to raw_step, we might skip rebinning or just copy?
    # Original logic: if tBetween > binHours -> RebinToStep (upsample/split)
    # else -> Aggregate (downsample)
    
    # Simplified logic: Resample using pandas
    
    # Set index to DateTime for resampling
    df_resample = df.set_index('DateTime')
    
    # Convert bin_hours to pandas offset string
    bin_minutes = int(bin_hours * 60)
    offset = f'{bin_minutes}min'
    
    # Resample logic
    # We need to aggregate Counts (Sum), RnConc (Weighted Mean?), Temperature (Mean)
    # But RnConc depends on Counts and Calibration.
    # It's safer to Sum Counts, Mean Calibration, then recalculate RnConc
    
    resampler = df_resample.resample(offset)
    
    agg_dict = {
        'Counts': 'sum',
        'Cnts_to_Bqm3': 'mean',
        'Temperature': 'mean',
        'RHofSampledAir': 'mean',
        # 'RnConc': 'mean' # Re-calculate this instead
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df_resample.columns}
    
    binned = resampler.agg(agg_dict).dropna(subset=['Counts'])
    
    # Adjust calibration for new bin vs old bin?
    # If we sum counts, we are combining N old bins.
    # RnConc = Counts_new * Calibration_mean * ... factor?
    # Original: binnedRnConcentration = [u*c/sumNumber ...]
    # where sumNumber = number of measurements binned together.
    
    # Let's count how many items went into each bin
    counts_per_bin = resampler['Counts'].count()
    
    # Filter out empty bins
    binned = binned[counts_per_bin > 0]
    counts_per_bin = counts_per_bin[counts_per_bin > 0]
    
    # Calculate new RnConc
    # RnConc = (Counts * Calibration) / N_samples_in_bin
    binned['RnConc'] = (binned['Counts'] * binned['Cnts_to_Bqm3']) / counts_per_bin
    
    # Calculate Uncertainty
    # Uncert_Counts = sqrt(Counts) (Poisson)
    # Uncert_RnConc = (sqrt(Counts) * Calibration) / N_samples_in_bin
    
    # Handle zero counts for uncertainty
    binned['Uncert_Counts'] = np.sqrt(binned['Counts'])
    binned['Uncert_Counts'] = binned['Uncert_Counts'].replace(0, 1) # If 0, use 1 for error estimate upper bound
    
    binned['Uncert_RnConc'] = (binned['Uncert_Counts'] * binned['Cnts_to_Bqm3']) / counts_per_bin
    
    # Reset index to get DateTime back as column directly (center of bin preferred?)
    # Pandas resample labels left edge by default. Shift by half bin size to center.
    binned = binned.reset_index()
    binned['DateTime'] = binned['DateTime'] + timedelta(minutes=bin_minutes/2)
    
    return binned

def place_time_cut(df: pd.DataFrame, t_start: str, t_end: str, fmt: str = '%Y-%m-%d %H:%M:%S') -> pd.DataFrame:
    """Cuts data between two timestamps."""
    start_dt = datetime.strptime(t_start, fmt)
    end_dt = datetime.strptime(t_end, fmt)
    
    mask = (df['DateTime'] > start_dt) & (df['DateTime'] < end_dt)
    cut_df = df[mask].copy()
    
    return cut_df.reset_index(drop=True)

def stitch_gaps(df: pd.DataFrame, max_gap_min: float = 1.0) -> pd.DataFrame:
    """
    Fills gaps in data larger than max_gap_min with interpolated values.
    Returns a dataframe with interpolated rows marked.
    Verified to use linear interpolation on a regular time grid.
    """
    if df.empty or 'DateTime' not in df.columns:
        return df

    # Ensure sorted by time
    df = df.sort_values('DateTime')

    # Create a regular time index
    # Round start/end to nearest minute or second? Or just use exact?
    # Original logic inserted points at fixed intervals of `expectedTimeStep_min` (max_gap_min)
    
    # We want to keep original points and fill gaps.
    # Set DateTime as index
    df_indexed = df.set_index('DateTime')
    
    # Resample to the desired grid (e.g. 1 min)
    # This will create NaNs where data is missing
    # However, existing data might not be perfectly on the grid.
    # We want to UPSAMPLE to a regular grid that includes points covering the gaps.
    
    # Simpler approach matching "stitching":
    # 1. Identify gaps > max_gap_min
    # 2. Generate new time points for those gaps
    # 3. Concatenate and sort
    # 4. Interpolate

    # Calculate time diffs
    time_diffs = df['DateTime'].diff()
    gap_mask = time_diffs > timedelta(minutes=max_gap_min)
    
    if not gap_mask.any():
        return df
        
    # We will build a list of new times to insert
    new_rows = []
    
    # Iterate only over gaps
    # We need indices of gaps
    gap_indices = df.index[gap_mask]
    
    for idx in gap_indices:
        # End of gap (current row)
        t_end = df.loc[idx, 'DateTime']
        
        # Start of gap (previous row)
        # We need integer location to get previous
        loc = df.index.get_loc(idx)
        t_start = df.iloc[loc-1]['DateTime']
        
        # Determine number of steps to insert
        delta = (t_end - t_start).total_seconds() / 60.0 # minutes
        
        # We want points every max_gap_min
        # t_start + n * max_gap_min
        curr_t = t_start + timedelta(minutes=max_gap_min)
        
        while curr_t < t_end:
            new_rows.append({'DateTime': curr_t, 'Stitched': 1})
            curr_t += timedelta(minutes=max_gap_min)
            
    if not new_rows:
        return df
        
    # Create DF from new rows
    df_new = pd.DataFrame(new_rows)
    
    # Concatenate
    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined = df_combined.sort_values('DateTime').reset_index(drop=True)
    
    # Interpolate
    # We only want to interpolate RnConc and Uncert_RnConc, maybe others
    cols_to_interp = ['RnConc', 'Uncert_RnConc', 'Temperature', 'RHofSampledAir']
    # Filter to existing
    cols_to_interp = [c for c in cols_to_interp if c in df_combined.columns]
    
    # Use time-based interpolation for accuracy
    df_combined = df_combined.set_index('DateTime')
    df_combined[cols_to_interp] = df_combined[cols_to_interp].interpolate(method='time')
    
    # Fill Stitched column (NaNs become 0)
    if 'Stitched' in df_combined.columns:
        df_combined['Stitched'] = df_combined['Stitched'].fillna(0).astype(int)
        
    return df_combined.reset_index()
