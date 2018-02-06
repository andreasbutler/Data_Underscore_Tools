import fit_tools as fitter
import image_tools as its
import deconv_tools
import os
import numpy as np
from matplotlib import pyplot as plt
from math import e
from scipy.constants import physical_constants as pcs
from scipy.misc import imsave

index_to_info_45ev_dict = {
    "11":{'temp': 30, 'mu':327},
    "16":{'temp': 50, 'mu':327},
    "20":{'temp': 70, 'mu':327},
    "24":{'temp': 90, 'mu':327},
    "29":{'temp': 110, 'mu':327},
    "33":{'temp': 130, 'mu':327},
    "37":{'temp': 150, 'mu':327},
    "41":{'temp': 170, 'mu':327},
    "44":{'temp': 190, 'mu':327},
    "48":{'temp': 210, 'mu':327},
    "52":{'temp': 230, 'mu':277},
    "56":{'temp': 260, 'mu':277},
    "60":{'temp': 285, 'mu':277}}

def fit_fermi_calib(data_file):
    folder_name = "calib_ff"
    fpath = os.path.join(folder_name, data_file)
    data = np.loadtxt(fpath)
    energy_data = [item[0] for item in data]
    density_data = [item[1] for item in data]
    guess = [40.8, 80, 0.05]
    optim = fitter.least_squares_optimize(fermi_func, guess, energy_data[155:], density_data[155:])
    print(optim)
    plt.plot(energy_data[:], fermi_func(energy_data[:], *optim), lw=2, c='r', label='Fermi function fit')
    plt.plot(energy_data[:], density_data[:], label="raw Fermi function data")
    plt.ylabel("Electron Density (normalized)")
    plt.xlabel("Energy (eV)")
    plt.legend()
    plt.show()


def fermi_func(E, mu, T, units_multiplier, offset):
    k_b = pcs["Boltzmann constant in eV/K"][0]*units_multiplier
    return 1/(e**((E-mu)/(k_b*T))+1) + offset


def deconv_full_spectra(filter_width, iterations, units_multiplier, folder_path, output_path):
    index_to_temp_45ev_dict = {"11":30 , "16":50, "20":70, "24":90, "29":110, "33":130, "37":150, "41":170,
    "44":190, "48":210, "52":230, "56":260, "60":285}
    filter = its.generate_gaussian_filter(filter_width)
    for f in os.listdir(folder_path):
        data = np.loadtxt(os.path.join(folder_path, f))
        data /= np.max(data)
        deconved_data = its.restoration.richardson_lucy(data, filter, iterations=iterations)
        index = f[2:-4]
        if "11" in f:
            plt.imshow(data)
            plt.xlabel("k")
            plt.ylabel("E")
            plt.figure()
            plt.imshow(deconved_data)
        if index in ["52","56","50"]:
            mu = 277
        else:
            mu = 327
        for E in range(len(deconved_data)):
            for k in range(len(deconved_data[0])):
                deconved_data[E][k] /= fermi_func(E, mu, index_to_temp_45ev_dict[index], units_multiplier, 0)
        # plt.figure()
        # plt.imshow(deconved_data[:-70,:])
        if "11" in f:
            plt.figure()
            plt.imshow(deconved_data[:-50,:])
            plt.xlabel("k")
            plt.ylabel("E")
        np.savetxt(os.path.join(output_path, "deconv_" + f), deconved_data)


def collect_columns(col_number, folder_path, output_path, average_width=0):
    plt.figure()
    for f in os.listdir(folder_path):
        data = np.loadtxt(os.path.join(folder_path, f))
        data /= np.max(data)
        data_T = data.transpose()
        col = data_T[col_number]
        for i in range(col_number - average_width, col_number + average_width + 1):
            if i != 0:
                col += data_T[i]
        col /= 1 + 2*average_width
        plt.figure()
        data_range = range(50, len(col[:-50]) + 50) if any(x in f for x in ["52", "56", "60"]) else range(len(col[:-50]))
        data_range_energy = [2*(E - 327) for E in data_range]
        # plt.plot(data_range_energy, col[:-50], label=str(index_to_temp_45ev_dict[f[2:-4]]) + "K")
        np.savetxt(os.path.join(output_path, "col" + str(col_number) + "_" + f), col)
    # plt.xlabel("E (meV)")
    # plt.ylabel("Electron density")


if __name__ == "__main__":

    # filt = its.generate_gaussian_filter(6.8)
    #
    # data = np.loadtxt("exported_waves\\JUNE_PGM_S9\\raw_data\\raw_spectrums\\gr48.txt")
    # data /= np.max(data)
    # dec_5 = its.restoration.richardson_lucy(data, filt, iterations=5)
    # dec_10 = its.restoration.richardson_lucy(data, filt, iterations=10)
    # dec_20 = its.restoration.richardson_lucy(data, filt, iterations=20)
    #
    # raw_curve = data.transpose()[66]
    # dec5_curve = dec_5.transpose()[66]
    # dec10_curve = dec_10.transpose()[66]
    # dec20_curve = dec_20.transpose()[66]
    #
    # for E in range(len(raw_curve)):
    #     dec_5[E] /= fermi_func(E, 327, index_to_temp_45ev_dict["48"], 0)
    #     dec_10[E] /= fermi_func(E, 327, index_to_temp_45ev_dict["48"], 0)
    #     dec_20[E] /= fermi_func(E, 327, index_to_temp_45ev_dict["48"], 0)
    #
    # data_range = range(len(raw_curve))[:-50]
    # data_range_energy = [2*(E - 327) for E in data_range]
    # plt.plot(data_range_energy, dec20_curve[:-50], label="20 iters")
    # plt.plot(data_range_energy, dec10_curve[:-50], label="10 iters")
    # plt.plot(data_range_energy, dec5_curve[:-50], label="5 iters")
    # plt.plot(data_range_energy, raw_curve[:-50], label="raw data")
    #
    # plt.xlabel("Energy (meV)")
    # plt.ylabel("Electron density")
    # plt.title("150K")

    input_dir = "exported_waves\\JUNE_PGM_S9\\raw_data\\raw_spectrums\\"
    int_dir = "exported_waves\\JUNE_PGM_S9\\raw_data\\deconved_spectrums\\"
    output_dir = "exported_waves\\JUNE_PGM_S9\\raw_data\\deconved_columns\\col_66\\"

    deconv_tools.deconv_full_spectra(
        filter_width=6.8,
        iterations=5,
        index_to_info_dict=index_to_info_45ev_dict,
        units_multiplier=2000,
        folder_path=input_dir,
        output_path=int_dir,
        save=True,
        plot=False)

    deconv_tools.collect_deconved_columns(
        col_number=66,
        index_to_info_dict=index_to_info_45ev_dict,
        folder_path=int_dir,
        output_path=output_dir,
        lower_cutoff=50,
        upper_cutoff=-60,
        average_width=0,
        save=True,
        plot=True)

    # for file in os.listdir(output_dir):
    #     print(file)
    #     data = np.loadtxt(os.path.join(output_dir, file))[30:-60]
    #     plt.figure()
    #     plt.plot(range(len(data)), data)

    # if index in ["52","56","50"]:
    #     mu = 277
    # else:
    #     mu = 327
    # for E in range(len(deconved_data)):
    #     for k in range(len(deconved_data[0])):
    #         deconved_data[E][k] /= fermi_func(E, mu, index_to_temp_45ev_dict[index], 0)
    # # plt.figure()
    # # plt.imshow(deconved_data[:-70,:])
    # if "11" in f:
    #     plt.figure()
    #     plt.imshow(deconved_data[:-50,:])
    #     plt.xlabel("k")
    #     plt.ylabel("E")
    # np.savetxt(os.path.join(output_path, "deconv_" + f), deconved_data)

    plt.legend()
    plt.show()

