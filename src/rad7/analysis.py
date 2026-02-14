"""
Analysis module for Rad7 data.
Handles curve fitting and goodness of fit calculations.
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Any, Optional, Callable

def exponential_fit(t: np.ndarray, amp: float, t_char: float, baseline: float = 0) -> np.ndarray:
    """
    Exponential decay function with baseline.
    y = amp * exp(-t/t_char) + baseline
    """
    return amp * np.exp(-t/t_char) + baseline

def exponential_fit_zero_baseline(t: np.ndarray, amp: float, t_char: float) -> np.ndarray:
    """
    Exponential decay function with zero baseline.
    y = amp * exp(-t/t_char)
    """
    return exponential_fit(t, amp, t_char, 0)

def goodness_of_fit(observed: np.ndarray, 
                    expected: np.ndarray, 
                    sigma: np.ndarray, 
                    num_fitted_params: int) -> Dict[str, float]:
    """
    Calculates Chi-squared, reduced Chi-squared, and p-value.
    """
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    sigma = np.asarray(sigma)
    
    # Avoid division by zero in sigma
    # Replace zeros with small number or handle appropriately
    # Standard practice if sigma is 0 is to ignore point or use weight 1?
    # Original code didn't handle sigma=0 explicitly here but processing ensured non-zero errors
    
    ndf = len(observed) - num_fitted_params
    
    if ndf <= 0:
        return {
            "chi_squared": np.nan,
            "ndf": ndf,
            "reduced_chi_squared": np.nan,
            "p_value": np.nan
        }

    residuals = observed - expected
    chi_squared = np.sum((residuals / sigma) ** 2)
    reduced_chi_squared = chi_squared / ndf
    p_value = 1 - stats.chi2.cdf(x=chi_squared, df=ndf)
    
    return {
        "chi_squared": chi_squared,
        "ndf": ndf,
        "reduced_chi_squared": reduced_chi_squared,
        "p_value": p_value
    }

def perform_fit(x_data: np.ndarray, 
                y_data: np.ndarray, 
                y_err: np.ndarray, 
                p0: Tuple[float, ...], 
                bounds: Optional[Tuple[Any, Any]] = None,
                fit_func: Callable = exponential_fit) -> Dict[str, Any]:
    """
    Performs curve fitting using scipy.optimize.curve_fit.
    """
    
    try:
        popt, pcov = curve_fit(
            fit_func, 
            xdata=x_data, 
            ydata=y_data, 
            sigma=y_err, 
            p0=p0, 
            bounds=bounds if bounds else (-np.inf, np.inf)
        )
        
        uncerts = np.sqrt(np.diag(pcov))
        expected = fit_func(x_data, *popt)
        
        gof = goodness_of_fit(y_data, expected, y_err, len(popt))
        
        return {
            "popt": popt,
            "uncerts": uncerts,
            "pcov": pcov,
            "gof": gof,
            "fitted_curve": expected
        }
        
    except Exception as e:
        print(f"Fit failed: {e}")
        return {}
