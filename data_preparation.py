# This method shows how the missing omni speed data is treated.
# It expects a csv file with a column called 'Date' with
# a recognisable date format.

import pandas as pd
import numpy as np

def load_data(data_file):
    data = pd.read_csv(data_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    data = data['2010-01-01 00:00:00':'2018-12-31 11:59:00']
    data = data.interpolate(method='time', limit=30)
    data = data.resample(f'30min').asfreq()
    data = data.fillna(np.nan)
    return data

# Once this method is used on the omni csv data, a list of timestamps
# from the index is taken (list(data.index)). These are used as the
# data points for the study. For each timestamp, the local database
# of solar images is checked to see if a suitable image exists- up
# to 30 minutes before the timestam[]