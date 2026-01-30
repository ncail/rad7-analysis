
# Imports
import pandas as pd


def parse_raw_data(r7raw_filepath):
    """
    Returns a dataframe of RAD7 data, parsed byte columns and a timestamp column.
    Some inconsequential columns are dropped.
    """
    columns = get_raw_data_column_names()
    df = pd.read_csv(r7raw_filepath, header=None, names=columns)
    print(df.head)

    # Consolidate datetime columns into one timestamp column.
    df_better_dt = convert_to_datetime(df.copy())

    # Drop uninteresting columns.
    df_pruned_cols = df_better_dt.copy()
    df_pruned_cols.drop(
        columns=[
            "Record Number",

        ],
        inplace=True
    )

    # Parse flags byte column and expand into more columns.
    df_expanded_cols = df_pruned_cols.copy()

    flags_expanded = df_expanded_cols['Flags Byte'].apply(parse_flags_byte).apply(pd.Series)
    df_expanded_cols = pd.concat([df_expanded_cols, flags_expanded], axis=1)

    # Parse units byte column and expand into more columns.
    units_expanded = df_expanded_cols['Units Byte'].apply(parse_units_byte).apply(pd.Series)
    df_expanded_cols = pd.concat([df_expanded_cols, units_expanded], axis=1)

    return df_expanded_cols
# End function.


def get_raw_data_column_names():
    # Each RAD7 cycle produces a record containing 23 comma-separated fields.
    # These columns are defined in RAD7 manual pg. 75.
    r7raw_columns = [
        "Record Number",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "Total Counts",
        "Live Time",
        "% of total counts in win. A",
        "% of total counts in win. B",
        "% of total counts in win. C",
        "% of total counts in win. D",
        "High Voltage Level",
        "High Voltage Duty Cycle",
        "Temperature",
        "Relative humidity of sampled air",
        "Leakage Current",
        "Battery Voltage",
        "Pump Current",
        "Flags Byte",
        "Radon concentration",
        "Radon concentration uncertainty",
        "Units Byte"
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
    r7raw_dataframe["Timestamp"] = pd.to_datetime(
        r7raw_dataframe[[
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute"
      ]]
    )
    r7raw_dataframe.drop(columns=["Year", "Month", "Day", "Hour", "Minute"], inplace=True)

    return r7raw_dataframe
# End function.


def parse_flags_byte(flags_byte: int) -> dict:
    """
    Parses the RAD7 flags byte (0-255) into individual components.

    Returns a dict with:
    - pump_state: 'Off', 'On', 'Timed', 'Grab'
    - thoron_on: True/False
    - measurement_type: 'Radon in Air', 'WAT-40', 'WAT250', 'Unknown'
    - auto_mode: True/False
    - sniff_mode: True/False
    """
    # Ensure it's 8-bit
    b = format(flags_byte, '08b')  # string '01010101'

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
        'pump_state': pump_state,
        'thoron_on': thoron_on,
        'measurement_type': measurement_type,
        'auto_mode': auto_mode,
        'sniff_mode': sniff_mode
    }
# End function.


def parse_units_byte(units_byte: int) -> dict:
    """
    Parses the RAD7 units byte (0-255) into human-readable units.

    Returns a dict with:
    - concentration_unit: 'Bq/m3', 'pCi/L', 'CPM', 'Total Counts'
    - temperature_unit: 'C' or 'F'
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
        'concentration_unit': concentration_unit,
        'temperature_unit': temperature_unit
    }
# End function.








