"""Calibrate X-ray spectra acquisition instrumentation with a linear fit
(first order approximation), comparing 60 kVp flat field acquisitions with
already calibrated Timepix4 and Amptek to be calibrated"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Timepix4_values = np.array([6.5, 13, 26.5, 32, 65]) # keV (from 60 kVp flat field spectrum)
Amptek_values = np.array([48, 110, 222.5, 272, 487]) # channels (from 60 kVp flat field spectrum)
sigma_Amptek = np.full(Amptek_values.shape, 5)
sigma_Timepix = np.full(Timepix4_values.shape, 0.5)

def simple_calibration(x, m, q):
    """Conversion from keV to channels as a linear model"""
    return m * x + q

# estimating initial parameters: expected approximately m=8, q=0
popt, pcov = curve_fit(simple_calibration, Timepix4_values, Amptek_values)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
print(f"calibration factor = {m_hat:.3f} +- {sigma_m:.3f}, offset {q_hat:.3f} +- {sigma_q:.3f}")

fig = plt.figure('Conversion from keV to channels')
plt.errorbar(Timepix4_values, Amptek_values, sigma_Amptek, sigma_Timepix, fmt='.')
x = np.linspace(2., 70., 120)
plt.plot(x, simple_calibration(x, m_hat, q_hat))
plt.xlabel('From Timepix4 keV')
plt.ylabel('From Amptek channels')
plt.grid(which='both', ls='dashed', color='gray')

chisq = (((Amptek_values - simple_calibration(Timepix4_values, *popt)) / sigma_Amptek)**2).sum()
print(f"chi square / dof = {chisq:.3f}/3")

plt.show()