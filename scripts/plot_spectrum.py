"""Plot spectra obtained as csv files from laboratory simulations and measurements"""

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

    args = parser.parse_args()

    n_files = len(args.file)
    _, axes = plt.subplots(n_files, 1, figsize=(8, 3 * n_files), sharex=True)

    if n_files == 1:
        axes = [axes]

    for ax, file_path in zip(axes, args.file):
        filename = os.path.splitext(os.path.basename(file_path))[0]

        if args.header:
            df = pd.read_csv(file_path, delimiter=args.delimiter)
        else:
            df = pd.read_csv(file_path, delimiter=args.delimiter, header=None)

        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        ax.plot(x, y, linewidth=1)
        ax.set_title(filename)
        ax.set_ylabel("Fluence")

        ax.grid(True)

    axes[-1].set_xlabel("Energy (keV)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
