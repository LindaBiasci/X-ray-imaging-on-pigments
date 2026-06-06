"""Plot spectra obtained as csv files from laboratory simulations and measurements:
'separate' mode is meant for simulations (i.e. plot incident, background and detail spectra),
'overlap' mode is meant for measurements (i.e. compare total attenuation of different samples);
default mode is 'both', which returns a separated plot and an overlapped plot (two figures)"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """Take file paths from terminal with argparse and make a plot for each one of them"""
    parser = argparse.ArgumentParser(description="Plot simulated X-ray spectra")

    parser.add_argument("file", nargs="+", help="Enter one or more csv file paths")
    parser.add_argument("--delimiter", default=";", help="Enter separator (default is ;)")
    parser.add_argument("--header", action="store_true", help="Does the csv file have a header?")
    parser.add_argument("--mode", choices=["separate", "overlap", "both"], default="both",
                        help="Choose plot mode: separate, overlap or both (default)")

    args = parser.parse_args()

    n_files = len(args.file)
    fig_sep, axes = None, None
    fig_overlap, ax_overlap = None, None

    # whenever separated plots are requested
    if args.mode in ["separate", "both"]:
        fig_sep, axes = plt.subplots(n_files, 1, figsize=(8, 3 * n_files), sharex=True)
        if n_files == 1:
            axes = [axes]

    # whenever overlapped plots are requested
    if args.mode in ["overlap", "both"]:
        fig_overlap, ax_overlap = plt.subplots(figsize=(10, 6))

    # read data from the given csv files
    for idx, file_path in enumerate(args.file):
        filename = os.path.splitext(os.path.basename(file_path))[0]

        if args.header:
            df = pd.read_csv(file_path, delimiter=args.delimiter)
        else:
            df = pd.read_csv(file_path, delimiter=args.delimiter, header=None)

        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        # for separated plots
        if axes is not None:
            ax = axes[idx]
            ax.plot(x, y, linewidth=1)
            ax.set_title(filename)
            ax.set_ylabel("Fluence")
            ax.grid(True)

        # for overlapped plots
        if ax_overlap is not None:
            ax_overlap.plot(x, y, linewidth=1.2, label=filename)

    # final configurations

    if axes is not None:
        axes[-1].set_xlabel("Energy (keV)")

    if ax_overlap is not None:
        ax_overlap.set_title("Overlapped X-ray Spectra")
        ax_overlap.set_xlabel("Energy (keV)")
        ax_overlap.set_ylabel("Total attenuation")
        ax_overlap.grid(True, which="both", ls=":")
        ax_overlap.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
