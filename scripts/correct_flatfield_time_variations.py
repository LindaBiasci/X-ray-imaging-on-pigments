"""Post-analyses correction of flat field energy shifts and FWHM variations with time"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Scaling of first column, i.e. slices")

    parser.add_argument("file1", help="First flat field")
    parser.add_argument("file2", help="Last flat field")
    parser.add_argument("--sep", default=",",
                            help="Enter csv separator (default is ',')")
    # a = 1 for first W peak, first flat field (slice 59)
    parser.add_argument("-a1", type=float, required=True,
                        help="Energy scaling factor, first W peak, first flat field")
    # b = 0.9853 for second W peak, first flat field
    parser.add_argument("-b1", type=float, required=True,
                            help="Energy scaling factor, second W peak, first flat field")
    # a = 0.9672 for first W peak, last flat field
    parser.add_argument("-a2", type=float, required=True,
                        help="Energy scaling factor, first W peak, last flat field")
    # b = 0.9710 for second W peak, last flat field
    parser.add_argument("-b2", type=float, required=True,
                            help="Energy scaling factor, second W peak, last flat field")

    args = parser.parse_args()

    # Read csv files
    df1 = pd.read_csv(args.file1, sep=args.sep)
    df2 = pd.read_csv(args.file2, sep=args.sep)

    # Apply energy scaling 
    x1 = df1.iloc[:, 0].astype(float).copy()
    x1.iloc[:59] *= args.a1
    x1.iloc[59:] *=args.b1
    y1 = df1.iloc[:, 1]

    x2 = df2.iloc[:, 0].astype(float).copy()
    x2.iloc[:61] *= args.a2
    x2.iloc[61:] *= args.b2
    y2 = df2.iloc[:, 1]

    # Plot results
    plt.figure(figsize=(8, 6))

    plt.plot(x1, y1, label=args.file1, linewidth=2)
    plt.plot(x2, y2, label=args.file2, linewidth=2)

    plt.xlabel("Scaled slices")
    plt.ylabel("Counts")
    plt.title("Flat field comparison after energy scaling")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
