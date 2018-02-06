import fit_tools as fitter
import image_tools as its
import function_tools
import os
import numpy as np
from matplotlib import pyplot as plt
from math import e
from scipy.constants import physical_constants as pcs


def deconv_full_spectra(
        filter_width,
        iterations,
        index_to_info_dict,
        units_multiplier,
        folder_path,
        output_path,
        save=True,
        plot=False):
    filt = its.generate_gaussian_filter(filter_width)
    for f in os.listdir(folder_path):
        data = np.loadtxt(os.path.join(folder_path, f))
        data /= np.max(data)
        deconved_data = its.restoration.richardson_lucy(data, filt, iterations=iterations)
        index = f[2:-4]
        if index in index_to_info_dict.keys():
            temp = index_to_info_dict[index]['temp']
            mu = index_to_info_dict[index]['mu']
            for E in range(len(deconved_data)):
                for k in range(len(deconved_data[0])):
                    deconved_data[E][k] /= function_tools.fermi_func(E, mu, temp, units_multiplier, 0)
            if save:
                np.savetxt(os.path.join(output_path, f[:-4] + "_deconv.txt"), deconved_data)
            if plot:
                plt.figure()
                plt.imshow(deconved_data)
        else:
            print("index " + str(index) + " not found in index to info dictionary.")


def collect_deconved_columns(
        col_number,
        index_to_info_dict,
        folder_path,
        output_path,
        average_width=0,
        lower_cutoff=None,
        upper_cutoff=None,
        save=True,
        plot=False):
    for f in os.listdir(folder_path):
        index = f[2:4]
        if index in index_to_info_dict.keys():
            data = np.loadtxt(os.path.join(folder_path, f))
            data_T = data.transpose()
            col = data_T[col_number][lower_cutoff:upper_cutoff]

            ##SMOOTHING
            for i in range(col_number - average_width, col_number + average_width + 1):
                if i != col_number:
                    print("hey")
                    col += data_T[i][lower_cutoff:upper_cutoff]
            col /= (1 + 2*average_width)

            mu = index_to_info_dict[index]['mu'] - lower_cutoff
            data_range_energy = [2*(E - mu) for E in range(len(col))]
            if save:
                np.savetxt(os.path.join(output_path, f[:-4] + "_col" + str(col_number) + ".txt"), col)
            if plot:
                #plt.figure()
                plt.plot(data_range_energy, col)
        else:
            print("Index " + str(index) + " not found in index to info dictionary.")
