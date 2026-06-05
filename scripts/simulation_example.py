"""Example program: a simple simulation of X-ray imaging"""

import numpy as np
import spekpy as sp
import matplotlib.pyplot as plt

"""Spectrum simulation: emitted from an X-ray tube, transmitted by a sample"""

# Set parameters:
# Spectrum configuration: maximum photons energy (tube's tension), anodic angle (degrees), anode's material
# Filter configuration: filter material, filter thickness (mm)
spectra = [
    {"kvp": 50, "th": 10, "targ": "Ag", "filter_material": "Al", "filter_thickness": 2.5, "label": "Ag 50 kVp"},
    {"kvp": 100, "th": 15, "targ": "W", "filter_material": "Cu", "filter_thickness": 0.1, "label": "W 100 kVp"},
    {"kvp": 45, "th": 15, "targ": "Cu", "filter_material": "Al", "filter_thickness": 2.0, "label": "Cu 45 kVp"}
]

sample_thickness = 5.0  # mm

def attenuation_coefficient(E):
    """A made-up attenuation law: the actual attenuation coefficient at fixed energy depends on the material"""
    return 10 / (E + 5)

"""Detector simulation: efficiency, energy resolution, SNR estimation"""

def detector_efficiency(E):
    """A made-up energy-dependent (keV) efficiency model"""
    return np.exp(-E / 50)

def energy_resolution_sigma(E):
    """A made-up energy resolution (FWHM-like behaviour converted to sigma)"""
    return 0.1 * np.sqrt(E) #keV

def apply_detector_response(energies, spectrum):
    """Applies efficiency and energy smearing, returns the detected spectrum"""
    energies = np.asarray(energies)
    spectrum = np.asarray(spectrum)

    # Efficiency
    eff = detector_efficiency(energies)
    detected = spectrum * eff

    # Energy smearing (Gaussian convolution approximation)
    smeared = np.zeros_like(detected)

    for i, E_in in enumerate(energies):
        sigma = energy_resolution_sigma(E_in)

        # Gaussian weights over all energies
        weights = np.exp(-(energies - E_in)**2 / (2 * sigma**2))
        weights /= np.sum(weights)

        # Resembling a gaussian convolution
        smeared[i] = np.sum(detected * weights)

    return smeared

# Create subplots
fig, axes = plt.subplots(1, len(spectra), figsize=(18, 5))

# Generate and plot spectra
for i, cfg in enumerate(spectra):

    # Create spectrum
    s = sp.Spek(kvp=cfg["kvp"], th=cfg["th"], targ=cfg["targ"])

    # Add filter
    s.filter(cfg["filter_material"], cfg["filter_thickness"])

    # Get emitted spectrum
    energies, fluence = s.get_spectrum()
    energies = np.asarray(energies)
    fluence = np.asarray(fluence)

    # Compute attenuation (Lambert-Beer law) and detection
    mu = attenuation_coefficient(energies)
    transmitted = fluence * np.exp(-mu * sample_thickness)
    detected = apply_detector_response(energies, transmitted)

    # A simple SNR estimation
    signal = np.sum(detected)
    noise = np.sqrt(signal + 1e-9)  # shot noise approximation
    snr = signal / noise

    # Plot emitted, transmitted and detected spectrum
    axes[i].plot(energies, fluence, label='Emitted spectrum')
    axes[i].plot(energies, transmitted, label='Transmitted spectrum')
    axes[i].plot(energies, detected, label='Detected spectrum')

    axes[i].set_title(f"{cfg['label']} | SNR={snr:.2f}")
    axes[i].set_xlabel("Energy [keV]")
    axes[i].set_ylabel("Fluence")
    axes[i].grid(True)
    axes[i].legend()

plt.tight_layout()
plt.show()
