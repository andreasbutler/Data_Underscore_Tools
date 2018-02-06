import numpy as np
from math import e, sqrt, pi
import plot_tools as plotter
from scipy.ndimage.filters import gaussian_filter
import fit_tools as fitter
from math import ceil
from scipy.constants import physical_constants as pcs
from matplotlib import pyplot as plt


def custom_gaussian_filter(width, step):
    g = lambda x: (1/(width*sqrt(2*pi)))*e**(-(x*x)/(2*width*width))
    filt_range = np.linspace(-3*ceil(width), 3*ceil(width), 6*ceil(width)/step + (1 if 6*ceil(width)/step % 2 == 0 else 0))
    gauss_filt = np.array([g(d) for d in filt_range])
    return gauss_filt, filt_range


def gaussian_convolve(data, step, width):
    f, frange = custom_gaussian_filter(width, step)
    half_length = int((len(f) - 1)/2)
    prepend = np.array([data[0]]*half_length)
    postpend = np.array([data[len(data) - 1]]*half_length)
    op = np.concatenate((prepend, data, postpend))
    ret = np.ones((len(data)))
    for i in range(len(ret)):
        ret[i] = np.dot(f, op[i: i + 2*half_length + 1])
    return ret*step

def gaussian(x, height, center, width):
    return height*np.exp(-(x - center)**2/(2*width**2))


def fermi_func(E, mu, T, units_multiplier, offset):
    k_b = pcs["Boltzmann constant in eV/K"][0]*units_multiplier
    return 1/(e**((E-mu)/(k_b*T))+1) + offset


def gaussian_convolution_1d(data, width):
    filtered = gaussian_filter(data, width, mode="nearest")
    return filtered


def convolution_least_squares(data, optimal_data, data_step, width_range, width_step):
    best_w = width_range[0]
    min_squares = 1000000000
    for w in np.linspace(width_range[0], width_range[1], (width_range[1] - width_range[0])/width_step):
        filt = gaussian_convolution_1d(data, w/data_step)
        diff = sum((filt - optimal_data)**2)
        if diff < min_squares:
            best_w = w
            min_squares = diff
    return best_w, min_squares


def custom_convolution_least_squares(data, optimal_data, data_step, width_range, width_step):
    best_w = width_range[0]
    min_squares = 1000000000
    for w in np.linspace(width_range[0], width_range[1], (width_range[1] - width_range[0])/width_step):
        filt = gaussian_convolve(data, data_step, w)
        diff = sum((filt - optimal_data)**2)
        if diff < min_squares:
            best_w = w
            min_squares = diff
    return best_w, min_squares


def calibration_proof(target_w):
    data_range, step = np.linspace(-0.01, 0.01, 1000, retstep=True)
    data = np.array([fermi_dirac(data_pnt, 0.000861733034) for data_pnt in data_range])
    plotter.simple_scatter(data_range, data)
    plotter.ylim(max=1.25, min=-0.25)
    filt = gaussian_convolution_1d(data, target_w)
    plotter.simple_scatter(data_range, filt)
    plotter.ylim(min=-0.25, max=1.25)
    plotter.simple_scatter(data_range, data - filt)
    plotter.ylim(min=-0.001, max=0.001)
    best_w, min_squares = convolution_least_squares(data, filt, step, [0, 3], 0.01)
    print(best_w, min_squares)
    plotter.show()


def match_fermi_dirac(real_temp, target_temp):
    kb = pcs["Boltzmann constant in eV/K"][0]*1000
    real_width = real_temp*kb
    data_range, step = np.linspace(-40, 40, 4000, retstep=True)
    data = np.array([fermi_dirac(data_pnt, real_width) for data_pnt in data_range])
    target_data = np.array([fermi_dirac(data_pnt, target_temp*kb) for data_pnt in data_range])
    plt.plot(data_range, data, label="10K Fermi function")
    plt.plot(data_range, target_data, label="47.6K Fermi function")
    best_w, least_squares = convolution_least_squares(data, target_data, step, [0.1, 10], 0.01)
    best_data = gaussian_convolution_1d(data, best_w/step)
    #best_w2, least_squares2 = custom_convolution_least_squares(data, target_data, step, [0.1, 10], 0.1)
    plt.plot(data_range, best_data, label="10K convolved")
    plt.ylabel("Electron density (normalized)")
    plt.xlabel("Energy (meV)")
    plt.legend()
    return best_w


if __name__ == "__main__":
    # kb = pcs["Boltzmann constant in eV/K"][0]*1000
    # real_width = 47.6*kb
    # data_range, step = np.linspace(-40, 40, 4000, retstep=True)
    # data = np.array([fermi_dirac(data_pnt, real_width) for data_pnt in data_range])
    print(match_fermi_dirac(10, 47.6))
    # data_range1, step = np.linspace(-40, 40, 4000, retstep=True)
    # kb = pcs["Boltzmann constant in eV/K"][0]
    # T = 10
    # data1 = np.array([fermi_dirac(E, kb*T*1000) for E in data_range1])
    # data2 = gaussian_convolution_1d(data1, 10/step)
    # data3 = np.array([fermi_dirac(E, 5) for E in data_range1])
    # r = gaussian_convolve(data1, data_range1, step, 10)
    # plotter.simple_scatter(data_range1, data1)
    # plotter.simple_scatter(data_range1, r)
    # plotter.simple_scatter(data_range1, data2)
    # plotter.simple_scatter(data_range1, data3)
    # plotter.simple_scatter(data_range1, r - data2)
    plotter.show()

