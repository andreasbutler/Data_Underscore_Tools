import os
import plot_tools as plotter

def save_plot(plot_dir, filename):
    plotter.save_plot(plot_dir, filename)

def save_params(fit_res, param_dir, filename):
    with open(os.path.join(param_dir,filename), 'w') as f:
        f.write(fit_res.fit_report())