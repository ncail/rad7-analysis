"""
Plotting module for Rad7 data.
Wrapper around matplotlib for standard plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

def plot_concentration(df: pd.DataFrame, 
                      x_col: str = 'DateTime', 
                      y_col: str = 'RnConc', 
                      y_err_col: str = 'Uncert_RnConc',
                      title: str = 'Radon Concentration',
                      save_path: Optional[str] = None):
    """
    Standard plot for Radon Concentration over time.
    """
    plt.figure(figsize=(12, 6))
    
    # Check if we can plot
    if df.empty:
        print("DataFrame is empty, nothing to plot.")
        return

    # Error flags might be zero or missing, handle gracefully
    y_err = df[y_err_col] if y_err_col in df.columns else None
    
    plt.errorbar(
        df[x_col], 
        df[y_col], 
        yerr=y_err, 
        fmt='o', 
        markersize=4, 
        capsize=3, 
        label='Data'
    )
    
    plt.xlabel(x_col)
    plt.ylabel(f'{y_col} [Bq/m$^3$]')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        # Create directory if needed
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        
    plt.show()

def plot_fit(x_data: np.ndarray, 
            y_data: np.ndarray, 
            y_fit: np.ndarray,
            y_err: Optional[np.ndarray] = None,
            x_label: str = 'Time',
            y_label: str = 'Concentration',
            title: str = 'Fit Result'):
    """
    Plots data vs fitted curve.
    """
    plt.figure(figsize=(10, 6))
    
    if y_err is not None:
        plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', label='Data', alpha=0.6)
    else:
        plt.scatter(x_data, y_data, label='Data', alpha=0.6)
        
    plt.plot(x_data, y_fit, 'r-', linewidth=2, label='Fit')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
