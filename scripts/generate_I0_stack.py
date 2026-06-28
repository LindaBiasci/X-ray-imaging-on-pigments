"""Generate a stack of 150 slices to be used as background in K-edge analyses of pigments"""

import tifffile
import argparse
import pandas as pd
import numpy as np


def main():

    # Setting 
    parser = argparse.ArgumentParser(description="Required arguments to generate I0 stack")
    parser.add_argument("-I0_up", "--input_upper_matrix", required=True,
                        help="Enter file path for I0 (e.g. wood), upper matrix")
    parser.add_argument("-I0_down", "--input_lower_matrix", required=True,
                        help="Enter file path for I0 (e.g. wood), lower matrix")
    parser.add_argument("--delimiter", default=",",
                        help="Enter separator (default is ,)")
    parser.add_argument("-o", "--output", default=None,
                        help="Enter file path to save the output tif data")

    # add n_slices basing on kVp
    parser.add_argument("-n", "--n_slices", required=True, type=int,
                        help="Enter n_slices")

    args = parser.parse_args()

    n_slices = args.n_slices
    height = 512
    half_height = 256
    width = 448

    # Read input csv files
    df_up = pd.read_csv(args.input_upper_matrix,
                        delimiter=args.delimiter)
    
    df_down = pd.read_csv(args.input_lower_matrix,
                          delimiter=args.delimiter)

    if len(df_up) != args.n_slices:
        raise ValueError(
            f"Upper CSV: expected {args.n_slices} rows, found {len(df_up)}")

    if len(df_down) != args.n_slices:
        raise ValueError(
            f"Lower CSV: expected {args.n_slices} rows, found {len(df_down)}")
    
    # Extract intensities
    values_up = df_up["Mean"].to_numpy(dtype=np.float32)
    values_down = df_down["Mean"].to_numpy(dtype=np.float32)

    # Generate stack
    stack = np.empty((args.n_slices, height, width), dtype=np.float32)
    stack[:, :half_height, :] = values_up[:, np.newaxis, np.newaxis]
    stack[:, half_height:, :] = values_down[:, np.newaxis, np.newaxis]

    print(f"Generated stack with shape {stack.shape}")

    tifffile.imwrite(args.output, stack, imagej=True)

    print(f"Stack saved to {args.output}")

if __name__ == "__main__":
    main()
