import numpy as np
from matplotlib import pyplot as plt
import os
import fitter as fit

def raw_plot(data):
    raw_figdir = figdir + "raw_slices\\"
    plt.figure()
    plt.plot(range(len(data)), data, lw=5, c='g', label='measurement'+str(file))

def gaussian_plot(data, filename):
    gauss_figdir = figdir + "gauss_fits\\"
    gauss_paramdir = paramdir + "gauss_fit_params\\"
    gauss_sep_figdir = figdir + "gauss_separate_peaks\\"
    gauss = fit.three_gaussians(data)
    plt.figure()
    plt.plot(range(len(data)), data, lw=5, c='g', label='measurement'+str(file))
    plt.plot(range(len(data)), gauss.best_fit)
    # plt.savefig(gauss_figdir  + filename[:4] +"fit3.png")
    # with open(gauss_paramdir + filename[:4] +"params3.txt", "w") as f:
    #     f.write(gauss.fit_report())
    # plt.plot(range(len(data)), [fit.gaussian(u, gauss.params['g1_height'], gauss.params['g1_center'], gauss.params['g1_sigma'], 0) for u in range(len(data))])
    # plt.plot(range(len(data)), [fit.gaussian(u, gauss.params['g2_height'], gauss.params['g2_center'], gauss.params['g2_sigma'], 0) for u in range(len(data))])
    # plt.plot(range(len(data)), [fit.gaussian(u, gauss.params['g3_height'], gauss.params['g3_center'], gauss.params['g3_sigma'], 0) for u in range(len(data))])
    #plt.plot(range(len(data)), [gauss.params['c'] for u in range(len(data))])
    # plt.savefig(gauss_sep_figdir + filename[:4] + "_1_gauss_sep_peaks.png")

def area_ratio_plot(data, filename):
    gauss_figdir = figdir + "gauss_fits\\"
    gauss_paramdir = paramdir + "gauss_fit_params\\"
    gauss_sep_figdir = figdir + "gauss_separate_peaks\\"
    gauss = fit.three_gaussians(data)
    (right_peak, left_peak) = (gauss.params["g3_amplitude"], gauss.params["g1_amplitude"]) \
        if (gauss.params["g1_center"] > gauss.params["g2_center"]) \
        else (gauss.params["g3_amplitude"], gauss.params["g2_amplitude"])
    # ratios.append(right_peak/left_peak)
    ratios.append(left_peak/right_peak)

def gaussian_plot_only(data):
    gauss = fit.three_gaussians(data)
    plt.figure()
    plt.plot(range(len(data)), data, lw=5, c='g', label='measurement'+str(file))
    plt.plot(range(len(data)), gauss.best_fit)
    print(gauss.params['slope'], gauss.params['intercept'])
    plt.plot(range(len(data)), [gauss.params['slope']*u + gauss.params['intercept'] for u in range(len(data))])
    plt.plot(range(len(data)), [fit.gaussian(u, gauss.params['g1_height'], gauss.params['g1_center'], gauss.params['g1_sigma'], 0) for u in range(len(data))])
    plt.plot(range(len(data)), [fit.gaussian(u, gauss.params['g2_height'], gauss.params['g2_center'], gauss.params['g2_sigma'], 0) for u in range(len(data))])
    plt.plot(range(len(data)), [fit.gaussian(u, gauss.params['g3_height'], gauss.params['g3_center'], gauss.params['g3_sigma'], 0) for u in range(len(data))])

def gaussian_plot_with_linear_save(data, filename):
    g_l_figdir = figdir + "gauss_linear_background_sep_peaks\\"
    g_l_fitdir = paramdir + "gauss_lin_back_params\\"
    lm = fit.linear_model(data[:40])
    gauss = fit.gaussians_with_linear(data, lm)
    plt.figure()
    plt.plot(range(len(data)), data, lw=5, c='g', label='measurement'+str(file))
    plt.plot(range(len(data)), gauss.best_fit)
    print(gauss.params['slope'], gauss.params['intercept'])
    plt.plot(range(len(data)), [gauss.params['slope']*u + gauss.params['intercept'] for u in range(len(data))])
    plt.plot(range(len(data)), [fit.gaussian(u, gauss.params['g1_height'], gauss.params['g1_center'], gauss.params['g1_sigma'], 0) for u in range(len(data))])
    plt.plot(range(len(data)), [fit.gaussian(u, gauss.params['g2_height'], gauss.params['g2_center'], gauss.params['g2_sigma'], 0) for u in range(len(data))])
    plt.plot(range(len(data)), [fit.gaussian(u, gauss.params['g3_height'], gauss.params['g3_center'], gauss.params['g3_sigma'], 0) for u in range(len(data))])
    with open(g_l_fitdir + filename + ".txt", "w") as f:
        f.write(gauss.fit_report())
    plt.savefig(g_l_figdir + filename + ".png")
    print(gauss.fit_report())

def simple_plot(data):
    data = data[:]
    plt.plot(range(len(data)), data)

def width_plot(rootdir):
    width_dir = rootdir + "\\width_data_col70\\"
    for subdir, dirs, files in os.walk(width_dir):
        for file in files:
            data = np.genfromtxt(os.path.join(subdir, file))
            plt.figure()
            plt.plot(range(len(data)), data)
    plt.show()

def width_gauss_fit_plot(rootdir, plot=False, save=False, collect_widths=False):
    width_dir = rootdir + "\\width_data_col70\\"
    width_fig_dir = figdir + "width_fit_figs\\"
    width_param_dir = paramdir + "width_params\\"
    widths = []
    temperatures = []
    for subdir, dirs, files in os.walk(width_dir):
        for file in files:
            data = np.genfromtxt(os.path.join(subdir, file))
            fitted = fit.three_gaussians(data)
            #print(fitted.params["g1_center"].value, fitted.params["g2_center"].value, fitted.params["g3_center"].value)
            if plot:
                plt.figure()
                plt.plot(range(len(data)), data)
                plt.plot(range(400), [fitted.eval(x=u) for u in range(400)])
                if save:
                    plt.savefig(width_fig_dir + file[:4] + "_width_fig_2.png")
                    with open(width_param_dir + file[:4] +"_width_params2.txt", "w") as f:
                        f.write(fitted.fit_report())
            if collect_widths:
                temperatures.append(temps[file[2:4]])
                widths.append(fitted.params["g3_fwhm"].value)
    if collect_widths:
        plt.figure()
        plt.plot(temperatures, widths)
        if save:
            with open(width_param_dir + "widths\\widths_2.txt", "w") as f:
                f.write(str(dict(zip(temperatures, widths))))
    plt.show()

def width_lorentz_fit_plot(rootdir):
    width_dir = rootdir + "\\width_data_col70\\"
    for subdir, dirs, files in os.walk(width_dir):
        for file in files:
            data = np.genfromtxt(os.path.join(subdir, file))
            fitted = fit.two_lortenzians(data)
            plt.figure()
            plt.plot(range(len(data)), data)
            plt.plot(range(len(data)), fitted.best_fit)
    plt.show()

def width_pv_fit_plot(rootdir):
    width_dir = rootdir + "\\width_data_col70\\"
    for subdir, dirs, files in os.walk(width_dir):
        for file in files:
            data = np.genfromtxt(os.path.join(subdir, file))
            fitted = fit.pseudo_voigt(data)
            plt.figure()
            plt.plot(range(len(data)), data)
            plt.plot(range(len(data)), fitted.best_fit)
    plt.show()


def plot_from_file(filepath):
    with open(filepath) as f:
        x_vs_y = eval(f.readline())
        x_vals = sorted(list(x_vs_y.keys()))
        y_vals = [x_vs_y[x] for x in x_vals]
        y_vals = [y/y_vals[0] for y in y_vals]
        plt.plot(x_vals, y_vals)
        plt.scatter(x_vals, y_vals)


if __name__ == "__main__":
    rootdir = 'C:\\Users\\Student\\Desktop\\Chatterjee\\ANDREAS-20170924T205427Z-001\\ANDREAS\\JUNE_PGM_Project\\exported_waves\\JUNE_PGM_S9'
    figdir = 'C:\\Users\\Student\\Desktop\\Chatterjee\\ANDREAS-20170924T205427Z-001\\ANDREAS\\JUNE_PGM_Project\\results\\fits\\figs\\'
    paramdir = 'C:\\Users\\Student\\Desktop\\Chatterjee\\ANDREAS-20170924T205427Z-001\\ANDREAS\\JUNE_PGM_Project\\results\\fits\\fit_params\\'
    ratios = []
    ratios2 = []
    temps = {"11":30 , "16":50, "20":70, "24":90, "29":110, "33":130, "37":150, "41":170, "44":190, "48":210, "52":230,
             "56":260, "60":285}
    xs = []
    width_gauss_fit_plot(rootdir, plot=True, save=False, collect_widths=True)
    file = ''
    # for subdir, dirs, files in os.walk(rootdir):
    #     for file in files:
    #         if(file[2:4] in ["29"]):
    #             data = np.genfromtxt(os.path.join(subdir, file))
    #             gaussian_plot_with_linear_save(data, "\\2_"+file[:-4])
                # gaussian_plot_only(data)
                # area_ratio_plot(data, file)
                # xs.append(temps[file[2:4]])
    # plt.plot(xs, ratios)
    # plt.scatter(xs, ratios)
    # plt.savefig(figdir + "area_ratios\\area_ratio4.png")
    # with open(figdir + "area_ratios\\area_ratio4_pnts.txt", "w") as f:
    #     f.write(str(dict(zip(xs,ratios))))
    # plt.plot(xs, ratios2)
    # plot_from_file(figdir + "area_ratios\\area_ratio_pnts_linear_background.txt")
    # plt.show()
