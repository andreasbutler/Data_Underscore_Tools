import fit_tools as fitter
import plot_tools as plotter
import save_tools as saver
import data_tools as datter
import numpy as np
import os

index_to_temp_45ev_dict = {"11":30 , "16":50, "20":70, "24":90, "29":110, "33":130, "37":150, "41":170,
    "44":190, "48":210, "52":230, "56":260, "60":285}

energy_to_temp_dict_dict = {45: index_to_temp_45ev_dict}

def area_ratios(fit_results, energy, num_index, denom_index):
    index_to_temp_dict = energy_to_temp_dict_dict[energy]
    temps = []
    ratios = []
    for fit_result in fit_results:
        temp_index = fit_results[fit_result][2][2:4]
        if temp_index != "56":
            centers = []
            for string in fit_result.params.keys():
                if "center" in string:
                    centers.append((fit_result.params[string], string[:-6]))
            centers = sorted(centers)
            ratio = fit_result.params[centers[num_index][1]+"amplitude"] / \
                fit_result.params[centers[denom_index][1]+"amplitude"]
            temps.append(index_to_temp_dict[temp_index])
            ratios.append(ratio)
    ratios = [r/max(ratios) for r in ratios]
    return np.array(temps), np.array(ratios)


def collect_widths(fit_results, energy, peak_index):
    index_to_temp_dict = energy_to_temp_dict_dict[energy]
    temps = []
    widths = []
    for fit_result in fit_results:
        temp_index = fit_results[fit_result][2][2:4]
        centers = []
        for string in fit_result.params.keys():
            if "center" in string:
                centers.append((fit_result.params[string], string[:-6]))
        centers = sorted(centers)
        width = fit_result.params[centers[peak_index][1]+"fwhm"]
        temps.append(index_to_temp_dict[temp_index])
        widths.append(width)
    return np.array(temps), np.array(widths)
        

def area_ratios_routine(save_fit_plots=False, save_params=False, save_area_ratios_plot=False):
    ##DEFINE ALL THE DIRECTORIES NEEDED FOR SAVING PLOTS/PARAMETERS
    data_dir = "C:\\Users\\Student\\Desktop\\Chatterjee\\ANDREAS-20170924T205427Z-001\\ANDREAS\\JUNE_PGM_Project\\" \
               "exported_waves\\JUNE_PGM_S9\\raw_data\\deconved_columns\\col_66\\"
    results_dir = "C:\\Users\\Student\\Desktop\\Chatterjee\\ANDREAS-20170924T205427Z-001\\ANDREAS\\JUNE_PGM_Project\\" \
               "results\\"
    gauss_fits_plot_dir = os.path.join(results_dir, "lr_fits\\45_eV\\figs\\lorentz_fits\\")
    area_ratios_plot_dir = os.path.join(results_dir, "lr_fits\\45_eV\\figs\\area_ratios\\")
    gauss_param_dir = os.path.join(results_dir, "lr_fits\\45_eV\\params\\lorentz_params\\")
    model_dir = "C:\\Users\\Student\\Desktop\\Chatterjee\\ANDREAS-20170924T205427Z-001\\ANDREAS\\JUNE_PGM_Project\\" \
                "model_frames\\max_response_frames_45eV\\"

    ##DEFINE THE MAPPING FROM DATA FILES TO MODEL DESCRIPTIONS
    data_to_info_dict = {
        "gr11_deconv_col66.txt": ["three_lorentzians.txt", 0, None],
        "gr16_deconv_col66.txt": ["three_lorentzians.txt", 0, None],
        "gr20_deconv_col66.txt": ["three_lorentzians.txt", 0, None],
        "gr24_deconv_col66.txt": ["three_lorentzians.txt", 0, None],
        "gr29_deconv_col66.txt": ["three_lorentzians.txt", 0, None],
        "gr33_deconv_col66.txt": ["three_lorentzians.txt", 0, None],
        "gr37_deconv_col66.txt": ["three_lorentzians.txt", 0, None],
        "gr41_deconv_col66.txt": ["three_lorentzians.txt", 0, None],
        "gr44_deconv_col66.txt": ["three_lorentzians.txt", 0, None],
        "gr48_deconv_col66.txt": ["three_lorentzians.txt", 0, None],
        "gr52_deconv_col66.txt": ["three_lorentzians_gr52.txt", 0, None],
        "gr56_deconv_col66.txt": ["three_lorentzians_gr52.txt", 0, None],
        "gr60_deconv_col66.txt": ["three_lorentzians_gr60.txt", 0, None]}

    ##FIT THE MODELS
    fit_results = fitter.fit_multiple(data_dir, model_dir, data_to_info_dict)

    ##PLOTS THE FITS
    plotter.plot_multiple_fits(
        fit_results=fit_results,
        plot_separate_peaks=True,
        save_plots=save_fit_plots,
        plot_dir=gauss_fits_plot_dir)

    ##SAVES PARAMETERS
    if save_params:
        for fit_result in fit_results:
            filename = fit_results[fit_result][2].rsplit('.')[0] + "_params.txt"
            saver.save_params(
                fit_res=fit_result,
                param_dir=gauss_param_dir,
                filename=filename)

    ##COLLECTS AREA RATIOS
    temps, ratios = area_ratios(
        fit_results=fit_results,
        energy=45,
        num_index=1,
        denom_index=2)

    ##PLOTS THE AREA RATIOS
    plotter.simple_scatter(temps, ratios)
    if save_area_ratios_plot:
        plotter.save_plot(
            plot_dir=area_ratios_plot_dir,
            filename="area_ratios_plot_lr")

    ##DISPLAY THE PLOTS
    plotter.show()


def widths_routine(save_fit_plots=False, save_params=False, save_widths_plot=False):
    ##DEFINE ALL THE DIRECTORIES NEEDED FOR SAVING PLOTS/PARAMETERS
    data_dir = "C:\\Users\\Student\\Desktop\\Chatterjee\\ANDREAS-20170924T205427Z-001\\ANDREAS\\JUNE_PGM_Project\\" \
               "exported_waves\\JUNE_PGM_S9\\width_data_col70\\"
    results_dir = "C:\\Users\\Student\\Desktop\\Chatterjee\\ANDREAS-20170924T205427Z-001\\ANDREAS\\JUNE_PGM_Project\\" \
               "results\\"
    width_gauss_fits_plot_dir = os.path.join(results_dir, "fits2\\figs\\width_gauss_fits\\")
    widths_plot_dir = os.path.join(results_dir, "fits2\\figs\\widths\\")
    width_gauss_param_dir = os.path.join(results_dir, "fits2\\params\\widths_gaussian_params\\")
    model_dir = "C:\\Users\\Student\\Desktop\\Chatterjee\\ANDREAS-20170924T205427Z-001\\ANDREAS\\JUNE_PGM_Project\\" \
                "model_frames\\off_response_frames\\"

    ##DEFINE THE MAPPING FROM DATA FILES TO MODEL DESCRIPTIONS
    data_to_info_dict = {
        "nr11COL70.txt": ["three_lorentzians.txt", 0, 7],
        "nr16COL70.txt": ["three_lorentzians.txt", 0, 8],
        "nr20COL70.txt": ["three_lorentzians.txt", 0, 10],
        "nr24COL70.txt": ["three_lorentzians.txt", 0, 15],
        "nr29COL70.txt": ["three_gaussians_nr29.txt", 0, 15],
        "nr33COL70.txt": ["three_lorentzians.txt", 0, 18],
        "nr37COL70.txt": ["three_lorentzians.txt", 0, 16],
        "nr41COL70.txt": ["three_gaussians_nr41.txt", 0, 17],
        "nr44COL70.txt": ["three_lorentzians.txt", 0, 13],
        "nr48COL70.txt": ["three_lorentzians.txt", 0, 8],
        "nr52COL70.txt": ["three_gaussians_nr52.txt", 49, None],
        "nr56COL70.txt": ["three_gaussians_nr56.txt", 49, None],
        "nr60COL70.txt": ["three_gaussians_nr60.txt", 49, None]}

    ##FIT THE MODELS
    fit_results = fitter.fit_multiple(data_dir, model_dir, data_to_info_dict, cut_off=10)

    ##PLOTS THE FITS
    plotter.plot_multiple_fits(
        fit_results=fit_results,
        plot_separate_peaks=True,
        save_plots=save_fit_plots,
        plot_dir=width_gauss_fits_plot_dir)

    ##SAVES PARAMETERS
    if save_params:
        for fit_result in fit_results:
            filename = fit_results[fit_result][2].rsplit('.')[0] + "_params.txt"
            saver.save_params(
                fit_res=fit_result,
                param_dir=width_gauss_param_dir,
                filename=filename)

    ##COLLECTS WIDTHS
    temps, widths = collect_widths(
        fit_results=fit_results,
        energy=45,
        peak_index=2)

    ##PLOTS THE WIDTHS
    plotter.simple_scatter(temps, widths)
    if save_widths_plot:
        plotter.save_plot(
            plot_dir=widths_plot_dir,
            filename="widths_plot")

    ##DISPLAY THE PLOTS
    plotter.show()


if __name__ == "__main__":
    area_ratios_routine(
        save_params=True,
        save_fit_plots=True,
        save_area_ratios_plot=True)


