"""

"""


# Imports
# RAD7 data processing packages
from src.utils import (
    parse_raw_data,
    get_unit
)

# Data manipulation
import pandas as pd

# Plotting
import matplotlib.pyplot as plt

def main():
    # Get r7raw data file to process.
    raw_data_path = "rad7_data/RAD7 04261 Data 2026-01-28.r7raw"
    rad7_run_df = parse_raw_data(raw_data_path)



if __name__ == '__main__':
    main()

