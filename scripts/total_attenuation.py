"""Compute and plot total attenuation -ln(I/I_0) for analysed pigments"""

import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def main():
    parser = argparse.ArgumentParser(description="Required arguments to compute -ln(I/I_0)")
    parser.add_argument("-I", "--input_I", required=True, 
                        help="Enter file path for I (transmitted radiation through sample)")
    parser.add_argument("-I0", "--input_I0", required=True,
                        help="Enter file path for I_0 (incident radiation)",)
    parser.add_argument("--delimiter", default=";",
                        help="Enter separator (default is ;)")
    parser.add_argument("-o", "--output", default=None,
                        help="Enter file path to save the output csv data (optional)")

    # add calibration parameters as optional (default values from fit)
    parser.add_argument("--m_cal", type=float, default=7.673,
                        help="Enter calibration factor (channels/keV)")
    parser.add_argument("--q_cal", type=float, default=10.996,
                        help="Enter calibration offset")

    args = parser.parse_args()

    try:
        data_I = np.loadtxt(args.input_I, delimiter=args.delimiter, skiprows=1)
    except FileNotFoundError:
        raise FileNotFoundError(f"Check file path: {args.input_I}")
    except (ValueError, TypeError):
        raise RuntimeError(f"Cannot read {args.input_I}, verify args in parser")

    try:
        data_I0 = np.loadtxt(args.input_I0, delimiter=args.delimiter, skiprows=1)
    except FileNotFoundError:
        raise FileNotFoundError(f"Check file path: {args.input_I0}")
    except (ValueError, TypeError):
        raise RuntimeError(f"Cannot read {args.input_I0}, verify args in parser")

    # extract columns from input files
    ch_I, I = data_I[:, 0], data_I[:, 1]
    ch_I0, I0 = data_I0[:, 0], data_I0[:, 1]

    # calibration: conversion from channels to keV
    x_keV = (ch_I - args.q_cal) / args.m_cal
    mask = x_keV <= 120.0
    x_keV = x_keV[mask]
    I = I[mask]
    I0 = I0[mask]

    # compute and plot total attenuation
    a = -np.log((I + np.finfo(float).eps) / (I0 + np.finfo(float).eps))

    # try to reduce noise
    window_size = 20 # number of channels to average on
    kernel = np.ones(window_size) / window_size
    a_smooth = np.convolve(a, kernel, mode='valid')
    x_keV_smooth = np.convolve(x_keV, kernel, mode='valid')

    if args.output:
        try:
            # save smoothed energies and attenuation values in a csv file
            output_data = np.column_stack((x_keV_smooth, a_smooth))
            header = "Energy_keV;Total_attenuation"
            np.savetxt(args.output, output_data, delimiter=args.delimiter, header=header, comments='')
            print(f"Saved data in {args.output}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the required file path: {args.output}")
        except OSError as e:
            raise RuntimeError(f"Failed data saving: {e}")

    plt.figure("Total attenuation", figsize=(10, 6))
    plt.plot(x_keV, a, color="lime", lw=0.6, label="Total attenuation")
    plt.plot(x_keV_smooth, a_smooth, color="darkred", lw=0.9, label="Total attenuation, smoothed")
    plt.title("Measured total attenuation (calibrated) with smoothing")
    plt.xlabel("Energy [keV]")
    plt.ylabel(r"$-\ln(I/I_0)$")
    plt.grid(True, which="both", ls=":")
    plt.show()

if __name__ == "__main__":
    main()
