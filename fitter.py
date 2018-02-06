from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize as op
from lmfit.models import PseudoVoigtModel, LinearModel, ConstantModel, VoigtModel, GaussianModel, LorentzianModel
from scipy import integrate
from math import pi as pi

def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset

def three_gaussians1(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, offset):
    return (gaussian(x, h1, c1, w1, offset=0) +
        gaussian(x, h2, c2, w2, offset=0) +
        gaussian(x, h3, c3, w3, offset=0) + offset)

def linear_model(data):
    lm = LinearModel()
    pars = lm.guess(data, x=np.array(range(len(data))))
    out = lm.fit(data, pars, x=np.array(range(len(data))))
    return out

def pseudo_voigt(data):
    pv_1 = PseudoVoigtModel(prefix="pv1_")
    pars1 = pv_1.guess(data, x=np.array(range(len(data))))
    pv_2 = PseudoVoigtModel(prefix="pv2_")
    pars2 = pv_2.guess(data, x=np.array(range(len(data))))
    pv_3 = PseudoVoigtModel(prefix="pv3_")
    pars3 = pv_3.guess(data, x=np.array(range(len(data))))
    background = ConstantModel()
    pars_background = background.guess(data, x=np.array(range(len(data))))
    mod = pv_1 + pv_2 + pv_3 + background
    pars1.update(pars2)
    pars1.update(pars3)
    pars1.update(pars_background)
    #pars1["pv1_sigma"].set()
    pars1.add("pv2_sigma", expr="pv1_sigma")
    pars1.add("pv3_sigma", expr="pv1_sigma")
    pars1["pv2_center"].set(245)
    pars1["pv3_center"].set(330)
    pars1["pv1_center"].set(300, min=280)
    pars1["pv2_center"].set(200, max=280)
    pars1["pv3_center"].set(200, max=280)
    # pars1["pv1_fraction"].set(0.5, min=0, max=1)
    # pars1["pv2_fraction"].set(0.5, min=0, max=1)
    # pars1["pv3_fraction"].set(0.5, min=0, max=1)
    # pars1['pv1_fwhm'].set(value=0.7, vary=True, expr='')
    # pars1['pv2_fwhm'].set(value=0.7, vary=True, expr='')
    # pars1['pv3_fwhm'].set(value=0.7, vary=True, expr='')
    out = mod.fit(data, pars1, x=np.array(range(len(data))))
    return out

def voigt(data):
    v_1 = VoigtModel(prefix="v1_")
    pars1 = v_1.guess(data, x=np.array(range(len(data))))
    v_2 = VoigtModel(prefix="v2_")
    pars2 = v_2.guess(data, x=np.array(range(len(data))))
    v_3 = VoigtModel(prefix="v3_")
    pars3 = v_3.guess(data, x=np.array(range(len(data))))
    background = ConstantModel()
    pars_background = background.guess(data, x=np.array(range(len(data))))
    mod = v_1 + v_2 + v_3 + background
    mod.make_params()
    pars1.update(pars2)
    pars1.update(pars3)
    pars1.update(pars_background)
    pars1.add("v2_sigma", expr="v1_sigma")
    pars1.add("v3_sigma", expr="v1_sigma")
    pars1["v1_center"].set(245, min=0, max=400)
    pars1["v2_center"].set(330, min=0, max=400)
    pars1["v3_center"].set(280, min=0, max=400)
    pars1["v1_amplitude"].set(245, min=0)
    pars1["v2_amplitude"].set(330, min=0)
    pars1["v3_amplitude"].set(280, min=0)
    # pars1["pv1_fraction"].set(0.5, min=0, max=1)
    # pars1["pv2_fraction"].set(0.5, min=0, max=1)
    # pars1["pv3_fraction"].set(0.5, min=0, max=1)
    pars1['v1_gamma'].set(value=30, vary=True, expr='', min=0, max=60)
    pars1['v2_gamma'].set(value=30, vary=True, expr='', min=0, max=60)
    pars1['v3_gamma'].set(value=30, vary=True, expr='', min=0, max=60)
    # pars1['v1_sigma'].set(value=30, vary=True, expr='', min=0, max=40)
    # pars1['v2_sigma'].set(value=30, vary=True, expr='', min=0, max=40)
    # pars1['v3_sigma'].set(value=30, vary=True, expr='', min=0, max=40)

    out = mod.fit(data, pars1, x=np.array(range(len(data))))
    return out

def three_gaussians(data):
    g1 = GaussianModel(prefix="g1_")
    pars = g1.guess(data, x=np.array(range(len(data))))
    g2 = GaussianModel(prefix="g2_")
    pars.update(g2.guess(data, x=np.array(range(len(data)))))
    g3 = GaussianModel(prefix="g3_")
    pars.update(g3.guess(data, x=np.array(range(len(data)))))
    background = ConstantModel()
    pars.update(background.guess(data, x=np.array(range(len(data)))))
    #pars['slope'].set(.001, min=0, max=.001)
    pars['g1_amplitude'].set(100, min=0)
    pars['g2_amplitude'].set(100, min=0)
    pars['g3_amplitude'].set(100, min=0)
    pars['g1_center'].set(250, min=0, max=280)
    pars['g2_center'].set(200, min=0, max=280)
    pars['g3_center'].set(350, min=250, max=320)
    pars['g1_sigma'].set(250, min=0, max=280)
    pars['g2_sigma'].set(200, min=0, max=280)
    pars['g3_sigma'].set()
    #pars['g3_height'].set(12, max=12)

    mod = g1 + g2 + g3 + background
    out = mod.fit(data, pars, x=np.array(range(len(data))))
    return out

def two_gaussians(data):
    g1 = GaussianModel(prefix="g1_")
    pars = g1.guess(data, x=np.array(range(len(data))))
    g2 = GaussianModel(prefix="g2_")
    pars.update(g2.guess(data, x=np.array(range(len(data)))))
    background = ConstantModel()
    pars.update(background.guess(data, x=np.array(range(len(data)))))
    #pars['slope'].set(.001, min=0, max=.001)
    pars['g1_amplitude'].set(100, min=0)
    pars['g2_amplitude'].set(100, min=0)
    # pars['g1_center'].set(250, min=0, max=280)
    # pars['g2_center'].set(200, min=280, max=400)

    mod = g1 + g2 + background
    out = mod.fit(data, pars, x=np.array(range(len(data))))
    return out

def two_lortenzians(data):
    g1 = LorentzianModel(prefix="g1_")
    pars = g1.guess(data, x=np.array(range(len(data))))
    g2 = LorentzianModel(prefix="g2_")
    pars.update(g2.guess(data, x=np.array(range(len(data)))))
    background = ConstantModel()
    pars.update(background.guess(data, x=np.array(range(len(data)))))
    #pars['slope'].set(.001, min=0, max=.001)
    pars['g1_amplitude'].set(100, min=0)
    pars['g2_amplitude'].set(100, min=0)
    # pars['g1_center'].set(250, min=0, max=280)
    # pars['g2_center'].set(200, min=280, max=400)

    mod = g1 + g2 + background
    out = mod.fit(data, pars, x=np.array(range(len(data))))
    return out

def gaussians_with_linear(data, lm):
    g1 = GaussianModel(prefix="g1_")
    pars = g1.guess(data, x=np.array(range(len(data))))
    g2 = GaussianModel(prefix="g2_")
    pars.update(g2.guess(data, x=np.array(range(len(data)))))
    g3 = GaussianModel(prefix="g3_")
    pars.update(g3.guess(data, x=np.array(range(len(data)))))
    background = LinearModel()
    pars.update(background.guess(data, x=np.array(range(len(data)))))
    #pars['slope'].set(.001, min=0, max=.001)
    pars['g1_center'].set(100, min=0, max=210)
    pars['g2_center'].set(200, min=0, max=280)
    pars['g3_center'].set(300, min=300, max=350)
    pars['g1_amplitude'].set(100, min=0)
    pars['g2_amplitude'].set(100, min=0)
    pars['g3_amplitude'].set(100, min=0)

    pars['slope'].set(lm.params['slope'], vary=False)
    pars['intercept'].set(lm.params['intercept'], vary=False)

    mod = g1 + g2 + g3 + background
    out = mod.fit(data, pars, x=np.array(range(len(data))))
    return out

def lorentzians(data):
    l1 = LorentzianModel(prefix="l1_")
    pars = l1.guess(data, x=np.array(range(len(data))))
    l2 = LorentzianModel(prefix="l2_")
    pars.update(l2.guess(data, x=np.array(range(len(data)))))
    l3 = LorentzianModel(prefix="l3_")
    pars.update(l3.guess(data, x=np.array(range(len(data)))))
    background = ConstantModel()
    pars.update(background.guess(data, x=np.array(range(len(data)))))
    pars['l1_amplitude'].set(100, min = 0)
    pars['l2_amplitude'].set(100, min = 0)
    pars['l3_amplitude'].set(100, min = 0)
    mod = l1 + l2 + l3 + background
    out = mod.fit(data, pars, x=np.array(range(len(data))))
    return out


def voigt_integral(x, sigma, gamma1, gamma2, gamma3, A1, A2, A3, c1, c2, c3):
    #f = lambda u : A*np.exp(-u**2/(2*sigma**2))/sigma/((2*pi)**(1/2))*gamma/(pi*((x-u-center)**2 - gamma**2))
    f2 = lambda y: np.exp(-y**2/(2*sigma))*(A1*gamma1/((y-x-c1)**2 + gamma1**2) +
                                            A2*gamma2/((y-x-c2)**2 + gamma2**2) +
                                            A3*gamma3/((x-y-c3)**2 + gamma3**2))
    #return integrate.quad(f, -np.inf, np.inf)[0]
    I = integrate.quad(f2, -np.inf, np.inf)[0]
    return I/(sigma*(2*pi*pi*pi)**(1/2))

def Voigt(a, u):
    I = integrate.quad(lambda y: np.exp(-y**2)/(a**2 + (u - y)**2),-np.inf, np.inf)[0]

    return (a/np.pi)*I

if __name__ == "__main__":
    data = np.genfromtxt("exported_waves\\JUNE_PGM_S9\\nr41COL66.txt")
    errfunc3 = lambda p, x, y: (three_gaussians1(x, *p) - y)**2
    guess3 = [2.3045, 189.42, 63.691, 2.695, 234.71, 19.508, 3.0745, 326.51, 26.07, 0]
    optim3, success = op.leastsq(errfunc3, guess3[:], args=(range(len(data)), data))
    # # errfunc = lambda p, x, y: (voigt_integral(x, *p) - y)**2
    # # guess = [36.9, 73.8, 73.8, 73.8, 444, 323, 18.2, 220, 240, 354]
    # # optim, success = op.leastsq(errfunc, guess[:], args=(np.asarray(range(len(data))), data))
    # res = pseudo_voigt(data)
    # #print(res.fit_report(min_correl=0.1))
    # plt.plot(range(len(data)), res.best_fit)
    # #plt.plot(range(len(data)), res.init_fit)
    plt.plot(range(len(data)), data, lw=5, c='g', label='measurement')
    # a = 0.1
    # u_range = np.linspace(-100,100,200)
    #
    # plt.plot(range(len(data)), [voigt_integral(u, 36.9, 73.8, 73.8, 73.8, 444, 323, 18.2, 233, 235, 354) for u in range(len(data))])
    # #plt.plot(u_range, [Voigt(a, u) for u in u_range])
    # #plt.show()
    plt.plot(range(len(data)), three_gaussians1(range(len(data)), *optim3), lw=3, c='b', label='fit of 3 Gaussians')
    plt.legend(loc='best')

    ##Data
    # data = np.genfromtxt("exported_waves\\JUNE_PGM_S9\\nr33COL66.txt")
    # plt.plot(range(len(data)), data, lw=5, c='g', label='measurement')

    ##Pseudo Voigt
    # res_pv = pseudo_voigt(data)
    # plt.plot(range(len(data)), res_pv.best_fit)
    # print(res_pv.fit_report(min_correl=0.1))

    ##Voigt
    # res_v = voigt(data)
    # plt.plot(range(len(data)), res_v.best_fit)
    # print(res_v.fit_report(min_correl=0.1))

    ##Gaussians
    # res_g = three_gaussians(data)
    # plt.plot(range(len(data)), res_g.best_fit)
    # print(res_g.fit_report(min_correl=0.1))

    ##Lorentzians
    # res_l = lorentzians(data)
    # plt.plot(range(len(data)), res_l.best_fit)
    # print(res_l.fit_report(min_correl=0.1))

    # plt.plot(range(len(data)), [gaussian(u, res_g.params['g1_height'], res_g.params['g1_center'], res_g.params['g1_sigma'], 0) for u in range(len(data))])
    # plt.plot(range(len(data)), [gaussian(u, res_g.params['g2_height'], res_g.params['g2_center'], res_g.params['g2_sigma'], 0) for u in range(len(data))])
    # plt.plot(range(len(data)), [gaussian(u, res_g.params['g3_height'], res_g.params['g3_center'], res_g.params['g3_sigma'], 0) for u in range(len(data))])
    # plt.plot(range(len(data)), [res_g.params['c'] for u in range(len(data))])

    plt.show()
