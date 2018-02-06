import numpy as np
from lmfit.models import PseudoVoigtModel, LinearModel, ConstantModel, VoigtModel, GaussianModel, LorentzianModel
import os
import fitter as fit
import data_tools as data_tools
import plot_tools as plotter
from scipy import optimize as op

model_name_to_constructor = {
    "gaussian": "GaussianModel",
    "constant": "ConstantModel",
    "lorentzian": "LorentzianModel",
    "linear": "LinearModel",
}

"""
The empty model is used to serve just as a base for a generic multi model fit. It is just the 0 function
"""
def empty_model(vary=False):
    model = ConstantModel(prefix="skeleton_")
    params = model.guess(data=np.array([0]), x=np.array([1]))
    params["skeleton_c"].set(0, vary=vary)
    return model, params

"""
A function that returns an arbitrary model of type model_name and its parameters, constrained as desired and initialized
to a guess of the given data
"""
def arbitrary_model(model_name, data_range, data, prefix="", constraints={}):
    m = globals()[model_name_to_constructor[model_name]](prefix=prefix)
    m_params = m.guess(data=data, x=data_range)
    for param in constraints.keys():
        if constraints[param][0]:
            m_params[prefix + param].set(constraints[param][0])
        m_params[prefix + param].set(min=constraints[param][1], max=constraints[param][2])
    return m, m_params


"""
Here is the fantastic multi model construction function. It lets us, in one fell swoop, set up a model with whatever
profile we so desire.
"""
def multi_model(data_range, data, models={}, vary_background=False):
    base_model, params = empty_model(vary=vary_background)
    for prefix in models.keys():
        model = models[prefix]
        mod, pars = arbitrary_model(
            model_name=model[0],
            data_range=data_range,
            data=data,
            prefix=prefix,
            constraints=model[1])
        base_model += mod
        params.update(pars)
    return base_model, params


def construct_model_from_file(data_range, data, model_dir, model_file, vary_background=False):
    model_str = None
    with open(os.path.join(model_dir, model_file), "r") as f:
        model_str = f.read()
    model, params = multi_model(
        data_range=data_range,
        data=data,
        vary_background=vary_background,
        models=eval(model_str))
    return model, params


def fit_multiple(data_dir, model_dir, data_to_info_dict, cut_off=None):
    results_dict = {}
    for data_file in data_to_info_dict.keys():
        data_range, data = data_tools.read_in_evenly_spaced_data(
            data_dir=data_dir,
            data_file=data_file,
            offset=data_to_info_dict[data_file][1])
        if data_to_info_dict[data_file][2]:
            data_range = data_range[:-data_to_info_dict[data_file][2]]
            data = data[:-data_to_info_dict[data_file][2]]
        model, params = construct_model_from_file(
            data_range=data_range,
            data=data,
            model_dir=model_dir,
            model_file=data_to_info_dict[data_file][0],
            vary_background=False)
        fit_res = model.fit(data, x=data_range, params=params)
        results_dict[fit_res] = [data_range, data, data_file]
    return results_dict


def least_squares_optimize(function, guess, data_range, data):
    errfunc = lambda p, x, y: (function(x, *p) - y)**2
    optim, success = op.leastsq(errfunc, guess[:], args=(data_range, data))
    if success:
        return optim
    else:
        return None

