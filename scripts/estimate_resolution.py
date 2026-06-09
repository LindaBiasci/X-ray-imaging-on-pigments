"""Read .bin file obtained from X-ray acquisition with Timepix4, plot spectrum and cluster size,
fit data to estimate experimental energy resolution"""

import numpy as np
import dask.array as da
import glob
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

folder = ["C:/Users/admin/Desktop/Tb_Fluorescence"]

En_list = []
X_list = []
Y_list = []
Size_list = []
ToT_list = []

for element in folder:

    En = np.memmap(os.path.join(element,"Energy.bin"), dtype=np.float16, mode='r')
    X = np.memmap(os.path.join(element,"X.bin"), dtype=np.int16, mode='r')
    Y = np.memmap(os.path.join(element,"Y.bin"), dtype=np.int16, mode='r')
    Size = np.memmap(os.path.join(element,"Size.bin"), dtype=np.int8, mode='r')
    ToT = np.memmap(os.path.join(element,"ToT.bin"), dtype=np.int16, mode='r')

    En_list.append(da.from_array(En, chunks=5_000_000))
    X_list.append(da.from_array(X, chunks=5_000_000))
    Y_list.append(da.from_array(Y, chunks=5_000_000))
    Size_list.append(da.from_array(Size, chunks=5_000_000))
    ToT_list.append(da.from_array(ToT, chunks=5_000_000))


En = da.concatenate(En_list)
X = da.concatenate(X_list)
Y = da.concatenate(Y_list)
Size = da.concatenate(Size_list)
ToT = da.concatenate(ToT_list)

print(f"Energy has {En.size} elements, cluster size has {Size.size} elements")
#print(En[:10])
#print(Size[:10])

# Energy bin definition 
bin_width = 0.5
num_bins = int(90 // bin_width) + 1
en_bins = np.linspace(0, 90, num_bins) 

# Cluster bin definition
size_bins = np.arange(0, 12) 

# Compute histograms
en_counts, en_edges = da.histogram(En, bins=en_bins)
size_counts, size_edges = da.histogram(Size, bins=size_bins)
en_counts, size_counts = da.compute(en_counts, size_counts)

def model_function(x, A, x0, sigma):
    """Gaussian function to model experimental energy resolution"""
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

# Bin centre definition
en_centers = (en_edges[:-1] + en_edges[1:]) / 2

# ROI selection (i.e. isolate a single peak)
roi_min, roi_max = 38.0, 56.0
mask_roi = (en_centers >= roi_min) & (en_centers <= roi_max)

x_fit = en_centers[mask_roi]
y_fit = en_counts[mask_roi]

# Estimate initial parameters
p0_A = np.max(y_fit)          
p0_x0 = 50.0                      
p0_sigma = 3.0                     

# Actual fit
try:
    popt, pcov = curve_fit(model_function, x_fit, y_fit, p0=[p0_A, p0_x0, p0_sigma])
    A_fit, x0_fit, sigma_fit = popt
    errors = np.sqrt(np.diag(pcov))
    
    # Compute FWHM
    fwhm = 2.35482 * sigma_fit
    fwhm_err = 2.35482 * errors[2]

    print(f"Peak energy: {x0_fit:.3f} ± {errors[1]:.3f} keV")
    print(f"Sigma: {sigma_fit:.3f} ± {errors[2]:.3f} keV")
    print(f"FWHM: {fwhm:.3f} ± {fwhm_err:.3f} keV")
    print(f"Peak amplitude: {A_fit:.1f} ± {errors[0]:.1f} counts")
    fit_success = True
except RuntimeError:
    print("\nFit failed, check p0")
    fit_success = False

# Plot data and fit 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.stairs(en_counts, en_edges, fill=True, color='royalblue', alpha=0.5, label="Data")
if fit_success:
    x_fine = np.linspace(roi_min, roi_max, 500)
    y_fine = model_function(x_fine, *popt)
    ax1.plot(x_fine, y_fine, color='crimson', lw=2.5, label=f"Gaussian centre: {x0_fit:.2f} keV\nFWHM: {fwhm:.2f} keV")
    ax1.legend(loc='upper right')

ax1.set_title("Energy spectrum with Gaussian Fit")
ax1.set_xlabel("Energy [keV]")
ax1.set_ylabel("Counts")
ax1.set_xlim(0, 90)
ax1.grid(True, alpha=0.3)

# Cluster size plot
valori_cluster = size_edges[:-1]
ax2.bar(valori_cluster, size_counts, width=0.6, color='darkorange', edgecolor='black', alpha=0.7)
ax2.set_title("Cluster size distribution")
ax2.set_xlabel("Cluster dimension (pixel)")
ax2.set_ylabel("Counts")
ax2.set_xticks(valori_cluster)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Filter a single cluster size value

# Chosen cluster size
target_size = 2
mask_single_cluster = (Size == target_size)
en_filtered = En[mask_single_cluster]

# Compute histogram with the chosen cluster size 
en_counts_filt, en_edges_filt = da.histogram(en_filtered, bins=en_bins)
en_counts_filt = en_counts_filt.compute()

# Fit on selected data 
en_centers_filt = (en_edges_filt[:-1] + en_edges_filt[1:]) / 2
roi_min_filt, roi_max_filt = 46.0, 55.0
mask_roi_filt = (en_centers_filt >= roi_min_filt) & (en_centers_filt <= roi_max_filt)

x_fit_filt = en_centers_filt[mask_roi_filt]
y_fit_filt = en_counts_filt[mask_roi_filt]

p0_A_filt = np.max(y_fit_filt)
p0_x0_filt = 49
p0_sigma_filt = 2.0

try:
    popt_filt, pcov_filt = curve_fit(model_function, x_fit_filt, y_fit_filt, p0=[p0_A_filt, p0_x0_filt, p0_sigma_filt])
    A_fit_f, x0_fit_f, sigma_fit_f = popt_filt
    errors_filt = np.sqrt(np.diag(pcov_filt))
    fwhm_filt = 2.35482 * sigma_fit_f
    
    print(f"Fit results for cluster size={target_size}:")
    print(f"Peak energy: {x0_fit_f:.3f} ± {errors_filt[1]:.3f} keV")
    print(f"FWHM: {fwhm_filt:.3f} ± {errors_filt[2]:.3f} keV")
    fit_filt_success = True
except (RuntimeError, TypeError):
    print(f"Fit failed for cluster size={target_size}")
    fit_filt_success = False

# Plot new figure
fig2, ax_filt = plt.subplots(figsize=(7, 5))
ax_filt.stairs(en_counts_filt, en_edges_filt, fill=True, color='forestgreen', alpha=0.5, label=f"Data (Size={target_size})")

if fit_filt_success:
    x_fine_filt = np.linspace(roi_min, roi_max, 500)
    y_fine_filt = model_function(x_fine_filt, *popt_filt)
    ax_filt.plot(x_fine_filt, y_fine_filt, color='darkred', lw=2.5, label=f"Gaussian centre: {x0_fit_f:.2f} keV\nFWHM: {fwhm_filt:.2f} keV")
    ax_filt.legend(loc='upper right')

ax_filt.set_title(f"Energy spectrum with cluster size {target_size}")
ax_filt.set_xlabel("Energy [keV]")
ax_filt.set_ylabel("Counts")
ax_filt.set_xlim(0, 90)
ax_filt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
