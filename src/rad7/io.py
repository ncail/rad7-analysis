"""
Input/Output module for Rad7 data.
"""

import os
import glob
import pandas as pd
from typing import List, Optional, Union
from pathlib import Path

from .raw_parser import parse_raw_data

def read_r7raw(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a single .r7raw file and returns it as a pandas DataFrame with appropriate headers.
    Uses raw_parser to handle initial parsing and byte decoding.
    
    Args:
        file_path (str or Path): Path to the .r7raw file.
        
    Returns:
        pd.DataFrame: DataFrame containing the raw data with standardized columns.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        # Use the parser from utils (now raw_parser)
        df = parse_raw_data(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()
        
    if df.empty:
        return df

    # Check for missing required columns used in processing
    required_cols = ['TotalCounts', 'PercentA', 'PercentC', 'PercentD', 'Temperature', 'RHofSampledAir', 'RnConc']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns in {file_path}: {missing}")

    return df

def load_data_from_directory(directory: Union[str, Path], extension: str = "*.r7raw") -> List[pd.DataFrame]:
    """
    Loads all files with the given extension from a directory.
    
    Args:
        directory (str or Path): Directory to search.
        extension (str): File glob pattern. Defaults to "*.r7raw".
        
    Returns:
        List[pd.DataFrame]: List of DataFrames.
    """
    directory = Path(directory)
    if not directory.exists():
         raise FileNotFoundError(f"Directory not found: {directory}")
         
    files = list(directory.glob(extension))
    if not files:
        print(f"No files matching {extension} found in {directory}")
        return []
        
    print(f"Loading {len(files)} files from {directory}...")
    data_frames = []
    for file in files:
        try:
            df = read_r7raw(file)
            if not df.empty:
                data_frames.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    return data_frames
