#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Script to visualize variable's distributions.
"""


import argparse

import matplotlib.pyplot as plt
import pandas as pd
from brainccpy.Clustering.utils import (plot_dist,
                                        remove_nans)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_df',
                   help='Input dataframe (.xlsx)')
    p.add_argument('--intervals', nargs='+', type=int,
                   help='Intervals of variables to plot together.'
                        'Ex: --intervals 1 4 will plot variables 1-4.')
    p.add_argument('--output', required=False,
                   help='Output directory and filename.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = pd.read_excel(args.in_df)
    df = remove_nans(df)

    # Plot distribution for all intervals.
    g = plot_dist(df, args.intervals[0], args.intervals[1])
    plt.savefig(f'{args.output}', format='png')


if __name__ == '__main__':
    main()
