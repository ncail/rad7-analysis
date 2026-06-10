"""
This module is for parsing raw RAD7 data files (`.r7raw` files) which have a standard format defined by the RAD7 manual (page 75).
"""


# Imports
import pandas as pd
from typing import Callable


def parse_raw_data(r7raw_filepath):
    """
    Returns a dataframe of RAD7 data, parsed byte columns and a timestamp column.
    Some inconsequential columns are dropped.
    """
    columns = get_raw_data_column_names()
    df = pd.read_csv(r7raw_filepath, header=None, names=columns)
    # print(df.head)

    # Convert Year to 4 digits (assume 2000+)
    df["Year"] = df["Year"].apply(lambda x: x + 2000)

    # Consolidate datetime columns into one timestamp column.
    df_better_dt = convert_to_datetime(df.copy())

    # Drop uninteresting columns.
    df_pruned_cols = df_better_dt.copy()
    df_pruned_cols.drop(
        columns=[
            # "recordNumber",
            # Others?
        ],
        inplace=True
    )

    # Parse flags byte column and expand into more columns.
    df_expanded_cols = df_pruned_cols.copy()
    df_expanded_cols = expand_byte_column(df_expanded_cols, "FlagsByte")

    # Parse units byte column and expand into more columns.
    df_expanded_cols = expand_byte_column(df_expanded_cols, "UnitsByte")

    # Sort the columns.
    # Maybe.

    return df_expanded_cols
# End function.


def expand_byte_column(
    rad_df,
    parse_function: Callable,
    byte_column_name: str
):
    """
    Wrapper for parse_flags_byte() or parse_units_byte() for use with DataFrames.

    Params:
        rad_df (pd.DataFrame): RAD7 dataframe
        parse_function (function): Function to parse the byte column (e.g. parse_flags_byte or parse_units_byte)
        byte_column_name (str): Column name of the byte column
    
    Returns:
        pd.DataFrame: Dataframe with expanded byte columns
    """
    
    byte_expanded = rad_df[byte_column_name].apply(parse_function).apply(pd.Series)
    df_expanded_cols = pd.concat([rad_df, byte_expanded], axis=1)
    df_expanded_cols.drop(columns=[byte_column_name], inplace=True)
    
    return df_expanded_cols


def get_raw_data_column_names():
    # Each RAD7 cycle produces a record containing 23 comma-separated fields.
    # These columns are defined in RAD7 manual pg. 75.
    r7raw_columns = [
        "recordNumber",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "TotalCounts",
        "LiveTime",
        "PercentA",
        "PercentB",
        "PercentC",
        "PercentD",
        "HighVoltageLevel",
        "HighVoltageDutyCycle",
        "Temperature",
        "RHofSampledAir",
        "LeakageCurrent",
        "BatteryVoltage",
        "PumpCurrent",
        "FlagsByte",
        "RnConc",
        "Uncert_RnConc",
        "UnitsByte"
    ]
    return r7raw_columns
# End function.


def convert_to_datetime(r7raw_dataframe):
    """
    r7raw file contains columns 1-5 as year, month, day, hour, minute.
    Consolidates these into one timestamp and drops columns.
    Returns dataframe.

    :param r7raw_dataframe: Dataframe created by parse_raw_data()
    :return: Dataframe with modified columns (neatens datetime information)
    """
    r7raw_dataframe["DateTime"] = pd.to_datetime(
        r7raw_dataframe[[
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute"
        ]]
    )
    # Don't drop them if processing.py might need them? 
    # Actually processing.py logic was updated to use DateTime.
    # But let's check processing.py again - it relies on DateTime existence.
    r7raw_dataframe.drop(columns=["Year", "Month", "Day", "Hour", "Minute"], inplace=True)

    return r7raw_dataframe
# End function.


def parse_flags_byte(flags_byte: int) -> dict:
    """
    Parses the RAD7 flags byte (0-255) into individual components.

    Returns a dict with:
    - PumpState: 'Off', 'On', 'Timed', 'Grab'
    - ThoronOn: True/False
    - MeasurementType: 'Radon in Air', 'WAT-40', 'WAT250', 'Unknown'
    - AutoMode: True/False
    - SniffMode: True/False
    """
    # Ensure it's 8-bit
    b = format(int(flags_byte), '08b')  # string '01010101'

    # Pump state bits 0-1 (least significant)
    pump_bits = b[-2:]
    pump_map = {
        '00': 'Off',
        '01': 'On',
        '10': 'Timed',
        '11': 'Grab'
    }
    pump_state = pump_map.get(pump_bits, 'Unknown')

    # Bit 3 = Thoron on (count from 0 = LSB)
    thoron_on = bool(int(b[-5]))

    # Bits 4-5 = Measurement type
    meas_bits = b[-6:-4]  # bits 4 and 5
    meas_map = {
        '00': 'Radon in Air',
        '10': 'WAT-40',
        '11': 'WAT250'
    }
    measurement_type = meas_map.get(meas_bits, 'Unknown')

    # Bit 6 = Auto mode
    auto_mode = bool(int(b[-3]))

    # Bit 7 = Sniff mode
    sniff_mode = bool(int(b[-8]))

    return {
        'PumpState': pump_state,
        'ThoronOn': thoron_on,
        'MeasurementType': measurement_type,
        'AutoMode': auto_mode,
        'SniffMode': sniff_mode
    }
# End function.


def parse_units_byte(units_byte: int) -> dict:
    """
    Parses the RAD7 units byte (0-255) into human-readable units.

    Returns a dict with:
    - ConcentrationUnit: 'Bq/m3', 'pCi/L', 'CPM', 'Total Counts'
    - TemperatureUnit: 'C' or 'F'
    """
    # Ensure it's 8-bit
    b = format(units_byte, '08b')

    # Bits 0-1: concentration
    conc_bits = b[-2:]
    conc_map = {
        '01': 'Bq/m3',
        '11': 'pCi/L',
        '00': 'CPM',
        '10': 'Total Counts'
    }
    concentration_unit = conc_map.get(conc_bits, 'Unknown')

    # Bit 7 = temperature unit
    temperature_unit = 'C' if b[0] == '1' else 'F'

    return {
        'ConcentrationUnit': concentration_unit,
        'TemperatureUnit': temperature_unit
    }
# End function.


def get_unit(df, variable_name):
    """
    Check that the unit of variable is consistent for this rad7 dataframe and return it.
    Applies best to 'Radon concentration Unit' and 'Temperature Unit'.
    variable_name must be a column name of the dataframe.
    """
    units = df[variable_name].dropna().unique()

    if len(units) != 1:
        raise ValueError(f"Inconsistent units found: {units}")

    return units[0]







