import numpy as np
from matplotlib import pyplot as plt
import os

def plot_multiple_fits(fit_results, plot_separate_peaks=False, save_plots=False, plot_dir=None):
    for fit_res in fit_results.keys():
        data_range = fit_results[fit_res][0]
        data = fit_results[fit_res][1]

        plt.figure()
        plt.title(fit_results[fit_res][2])
        plt.plot(data_range, data)
        plt.plot(data_range, fit_res.best_fit)
        if plot_separate_peaks:
            res_dict = fit_res.eval_components(data=data_range)
            for peak in res_dict.keys():
                if type(res_dict[peak]) is np.ndarray:
                    plt.plot(data_range, res_dict[peak])
        if save_plots:
            save_plot(plot_dir, fit_results[fit_res][2].rsplit('.')[0])


def plot_trimmed(data_range, data, pre_trim=0, post_trim=0):
    plt.figure()
    if post_trim > 0:
        plt.plot(data_range[pre_trim:-post_trim], data[pre_trim:-post_trim])
    else:
        plt.plot(data_range[pre_trim:], data[pre_trim:])


def simple_plot(data_range, data):
    plt.figure()
    plt.plot(data_range, data, marker='o')


def simple_scatter(data_range, data):
    plt.figure()
    plt.scatter(data_range, data)

def ylim(min=None, max=None):
    if min: plt.ylim(ymin=min)
    if max: plt.ylim(ymax=max)

def figure():
    plt.figure()

def save_plot(plot_dir, filename):
    plt.savefig(os.path.join(plot_dir, filename))


def show():
    plt.show()
