"""Plot calibrated Amptek X-ray spectra"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

m_fit = 7.673
q_fit = 10.996

def main():
    parser = argparse.ArgumentParser(description="Required arguments to plot a calibrated spectrum")
    parser.add_argument("--delimiter", default=";", help="Enter separator (default is ;)")
    parser.add_argument("--header", action="store_true", help="Does the csv file have a header?")
    parser.add_argument("--input", required=True, help="Enter file path for counts file")
    parser.add_argument("--output", default=None, help="Enter file path to save the output csv data (optional)")
    args = parser.parse_args()

    # read csv file
    header_option = 0 if args.header else None
    names_option = None if args.header else ["channeld", "counts"]

    df = pd.read_csv(
        args.input,
        delimiter=args.delimiter,
        header=header_option,
        names=names_option)

    col_channels = df.columns[0]
    col_counts = df.columns[1]

    # invert calibration: keV = (channels - q) / m
    df["energy_keV"] = (df[col_channels] - q_fit) / m_fit

    # create and plot spectrum
    plt.figure(figsize=(10, 6))
    plt.step(
        df["energy_keV"],
        df[col_counts],
        where="mid",
        color="darkblue",
        linewidth=1)

    plt.xlim(0, 120)
    plt.title(f"Calibrated X-ray spectrum: {args.input}", fontsize=12)
    plt.xlabel("Energy (keV)", fontsize=12)
    plt.ylabel("Photon counts", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    # save calibrated spectrum in a csv file
    if args.output:
        output_csv = args.output
    else:
        base_name, _ = os.path.splitext(args.input)
        output_csv = f"{base_name}_calibrated.csv"

    df_output = df[["energy_keV", col_channels, col_counts]]
    df_output.to_csv(output_csv, sep=args.delimiter, index=False)
    print(f"Saved file: {output_csv}")

if __name__ == "__main__":
    main()
