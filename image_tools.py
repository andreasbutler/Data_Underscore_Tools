import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from skimage import restoration
from math import pi, sqrt, log
from scipy.signal import fftconvolve, convolve
import os
import fit_tools as fitter

###
# THERE IS AN OUTSTANDING ISSUE WITH THE EDGES OF THE LUCY RICHARDSON DECONVOLUTION - 50 CHANNEL PROBLEM, WHY?
###

def test_2D_lorentzian(width):
    side = np.linspace(-300, 300, 601)
    X, Y = np.meshgrid(side, side)
    # Z = np.array((width/(2*pi))/((width*width + X*X + Y*Y)**(3/2)))
    Z = np.array((width/2)/((width*width/4 + Y*Y)))
    img = np.reshape(Z, (np.unique(X).size, np.unique(Y).size))
    return img


def generate_gaussian_filter(width):
    side = np.linspace(-100,100,201)
    X, Y = np.meshgrid(side, side)
    Z = np.array(1/(width*sqrt(2*pi))*np.exp(-(Y*Y/(width*width))/(2)))
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            if j != 100:
                Z[i][j] = 0
    filt = np.reshape(Z, (np.unique(X).size, np.unique(Y).size))
    s = filt.sum()
    return filt/s


def test_gaussian_convolution(image, width):
    ret = gaussian_filter(image, [width, 0], mode='nearest')
    return ret


##This is not nice and general...
def find_num_iterations(original_profile, convolved_profile, filter, min_iters=1, max_iters=20):
    best_i = 0
    best_val = 1000000000000
    for i in range(min_iters, max_iters+1):
        deconvolved = restoration.richardson_lucy(convolved_profile, filter, iterations=i)
        diff = (deconvolved[60:-60] - original_profile[60:-60])**2
        if diff.sum() < best_val:
            best_i = i
            best_val = diff.sum()
    return best_i


def get_best_iterations(lorentzian_width, w2_range, max_iters):
    initial_lorentzian = test_2D_lorentzian(lorentzian_width)
    ratios = []
    iters = []
    for width in w2_range:
        ratios.append(width/lorentzian_width)
        convolved_lorentzian = test_gaussian_convolution(initial_lorentzian, width)
        filt = generate_gaussian_filter(width)
        best_i = rl_get_iterations(initial_lorentzian, filt, convolved_lorentzian, max_iters)
        iters.append(best_i)
    return ratios, iters


def get_best_iterations_sweep_lorentz(filt_width, w2_range, max_iters):
    filt = generate_gaussian_filter(filt_width)
    ratios = []
    iters = []
    for width in w2_range:
        ratios.append(filt_width/width)
        temp_lorentz = test_2D_lorentzian(width)
        convolved_lorentzian = test_gaussian_convolution(temp_lorentz, filt_width)
        best_i = rl_get_iterations(convolved_lorentzian, filt, temp_lorentz, max_iters)
        iters.append(best_i)
    return ratios, iters


def richardson_lucy_fwhm(image, psf, original_fwhm, original_data, iterations=50, clip=True):
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(image.shape + psf.shape)
    fft_time =  np.sum([n*np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]

    center = len(image)//2

    iters = []
    ratios = []

    for i in range(iterations):
        relative_blur = image / convolve_method(im_deconv, psf, 'same')
        im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')
        fwhm = extract_fwhm(im_deconv.transpose()[len(im_deconv)//2], center)
        iters.append(i)
        ratios.append(fwhm/original_fwhm)

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return iters, ratios


def extract_fwhm(data, peak):
    half_peak_val = data[peak]/2
    check = peak
    while data[check] > half_peak_val:
        check+=1
    return 2*(check-peak)


#original_profile,#  convolved_profile, filter, min_iters=1, max_iters=20
def rl_get_iterations(image, psf, original_profile, max_iters=50, clip=True):
    best_i = 0
    best_val = 1000000000000

    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(image.shape + psf.shape)
    fft_time =  np.sum([n*np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]

    for i in range(max_iters):
        relative_blur = image / convolve_method(im_deconv, psf, 'same')
        im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')
        diff = (im_deconv[60:-60] - original_profile[60:-60])**2
        if diff.sum() < best_val:
            best_i = i
            best_val = diff.sum()

    return best_i


def rl(image, psf, iterations=50, clip=True):
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(image.shape + psf.shape)
    fft_time =  np.sum([n*np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]

    for _ in range(iterations):
        relative_blur = image / convolve_method(im_deconv, psf, 'same')
        im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')

    return im_deconv


def testing_deconv_widths():
    img = test_2D_lorentzian(30)
    ret = test_gaussian_convolution(img, 15)
    filt = generate_gaussian_filter(15)

    original_fwhm = extract_fwhm(img.transpose()[len(img)//2], len(img)//2)
    conv_fwhm = extract_fwhm(ret.transpose()[len(ret)//2], len(ret)//2)
    print(original_fwhm, conv_fwhm)
    iters, ratios = richardson_lucy_fwhm(ret, filt, original_fwhm, img, 50)

    plt.figure()
    plt.scatter(iters, ratios)
    plt.show()

if __name__ == "__main__":

    # ratios = np.loadtxt("results\\LR_optim\\L_sweep\\ratios.txt")[4:]
    # iters = np.loadtxt("results\\LR_optim\\L_sweep\\iters.txt")[4:]
    # iters = np.log(iters)
    # plt.scatter(ratios, iters, label="semi-log plot of iterations versus ratio")
    # line = lambda x, m, b: m*x + b
    # optim = fitter.least_squares_optimize(line, [20, 0], ratios, iters)
    # plt.plot(np.linspace(0, 0.3, 50), line(np.linspace(0,0.3,50), *optim), label=r'$33.59 \frac{\sigma_{PSF}}{\Gamma_d} + 0.0150$', c='r')
    # plt.xlabel(r'$\frac{\sigma_{PSF}}{\Gamma_d}$')
    # plt.ylabel(r'$\log{\;(iterations)}$')
    # plt.legend()
    # print(optim)

    lor = test_2D_lorentzian(10)
    conved = test_gaussian_convolution(lor, 6.8)

    original_lorentz = lor.transpose()[len(lor)//2]
    conved_lorentz = conved.transpose()[len(conved)//2]
    original_fwhm = extract_fwhm(original_lorentz, len(original_lorentz)//2)

    print(original_fwhm)
    iters, ratios = richardson_lucy_fwhm(conved, generate_gaussian_filter(6.8), original_fwhm, lor)
    plt.scatter(iters, ratios)
    plt.xlabel("iterations")
    plt.ylabel(r'$\Gamma_{deconvolved}\;/\;\Gamma_{original}$')

    plt.show()