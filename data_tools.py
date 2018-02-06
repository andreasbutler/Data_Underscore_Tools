import numpy as np
import os

"""
This function reads in files assuming data points are evenly spaced from one another. It defaults to unit spacing and
beginning at 0, but it can be offset arbitrarily and the displacement between points can be scaled.
"""
def read_in_evenly_spaced_data(data_dir, data_file, offset=0, scale=1):
    data = np.genfromtxt(os.path.join(data_dir, data_file))
    data_range = np.array([offset + i*scale for i in range(len(data))])
    return data_range, data
