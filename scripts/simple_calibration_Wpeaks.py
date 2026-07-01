"""Calibrate X-ray spectra acquisition instrumentation with a linear fit (first order approximation)"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

W_peaks = np.array([57.981, 59.318, 67.244, 69.089]) # keV (K_alpha 1, 2, K_beta 1, 2)
spectrum_peaks = np.array([457, 465, 526, 542]) # from spectrum plot, channels
sigma_spec_peaks = np.full(spectrum_peaks.shape, 2)
sigma_W_peaks = np.full(W_peaks.shape, 0.001)

def simple_calibration(x, m, q):
    """Conversion from keV to channels as a linear model"""
    return m * x + q

# estimating initial parameters: expected approximately m=8, q=0
popt, pcov = curve_fit(simple_calibration, W_peaks, spectrum_peaks)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
print(f"calibration factor = {m_hat:.3f} +- {sigma_m:.3f}, offset {q_hat:.3f} +- {sigma_q:.3f}")

fig = plt.figure('Conversion from keV to channels')
plt.errorbar(W_peaks, spectrum_peaks, sigma_spec_peaks, sigma_W_peaks, fmt='o')
x = np.linspace(50., 80., 150)
plt.plot(x, simple_calibration(x, m_hat, q_hat))
plt.xlabel('W peaks [keV]')
plt.ylabel('Measured peaks (channels)')
plt.grid(which='both', ls='dashed', color='gray')

chisq = (((spectrum_peaks - simple_calibration(W_peaks, *popt)) / sigma_spec_peaks)**2).sum()
print(f"chi square / dof = {chisq:.3f}/2")

plt.show()