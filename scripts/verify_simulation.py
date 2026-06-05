"""Check simulation results of total X-ray attenuation"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

def main(bg_path, detail_path, FWHM_target, bin_width):
    """Compute total expected attenuation -ln(I/I_0),
    applying a gaussian convolution to simulate detector resolution"""
    try:
        df_bg = pd.read_csv(bg_path)
        df_det = pd.read_csv(detail_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find file; details: {e}") from e
    except Exception as e:
        raise ValueError(f"Could not read csv file; details: {e}") from e

    # Conversion: from FWHM to standard deviation
    sigma_kev = FWHM_target / (2 * np.sqrt(2 * np.log(2)))  
    sigma_pixels = sigma_kev / bin_width

    # Column extraction: energy, background fluence, detail fluence
    energy = np.array(df_bg.iloc[:, 0].values)
    I_0 = np.array(df_bg.iloc[:, 1].values)*0.01
    I = np.array(df_det.iloc[:, 1].values)*0.01
    I_0_conv = gaussian_filter1d(I_0, sigma=sigma_pixels)
    I_conv = gaussian_filter1d(I, sigma=sigma_pixels)

    # Compute total attenuation avoiding divergences, with and without gaussian convolution
    valid_ideal = (I_0 > 0) & (I > 0)
    valid_conv = (I_0_conv > 0) & (I_conv > 0)
    id_attenuation = -np.log(I[valid_ideal] / I_0[valid_ideal])
    conv_attenuation = -np.log(I_conv[valid_conv] / I_0_conv[valid_conv])

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(energy[valid_ideal], id_attenuation, label='Ideal attenuation spectrum', color='blue', linestyle='--')
    plt.plot(energy[valid_conv], conv_attenuation, label=f'Expected attenuation spectrum with FWHM = {FWHM_target} keV', color='red', linewidth=2, alpha=0.7)

    plt.title('Sample attenuation spectrum with detector resolution', fontsize=12)
    plt.xlabel('Energy (keV)', fontsize=11)
    plt.ylabel('Total attenuation $-\\ln(I/I_0) = \\mu \\cdot x$', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.ylim(-1, 25)
    plt.show()

if __name__ == '__main__':
    BG_PATH = Path(r"C:\Users\linda\Desktop\xray_visibility_sim\runs\20260529_160444_simulation\detected_spectrum_background.csv")
    D_PATH = Path(r"C:\Users\linda\Desktop\xray_visibility_sim\runs\20260529_160444_simulation\detected_spectrum_detail.csv")
    resolution_fwhm = 4.0 # keV
    energy_bin = 2.0 # keV
    main(BG_PATH, D_PATH, resolution_fwhm, energy_bin)
