#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to standardize each variable from a dataframe with a specific approach.
"""


import argparse

import pandas as pd
import numpy as np
from brainccpy.Clustering.utils import (remove_nans,
                                        quantile_transform,
                                        power_transform)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_df',
                   help='Input dataframe (.xlsx)')
    p.add_argument('out_df',
                   help='Output directory and filename (.xlsx).')
    p.add_argument('--dict', required=False,
                   help='Text file containing the type of standardization to apply'
                        'to each variables.')
    p.add_argument('--nb_quant', required=False, default=100, type=int,
                   help='Number of quantiles to separate data into when using quantiles transform.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    df = pd.read_excel(args.in_df)
    df = remove_nans(df)
    length = len(df)

    d = {}
    with open(f'{args.dict}') as f:
        for line in f:
            (key, val) = line.split()
            d[key] = val

    for a in d.keys():
        if d[f'{a}'] == 'quant':
            dt = quantile_transform(np.array(df.loc[:, f'{a}']).reshape((length, 1)),
                                    args.nb_quant)
            df.loc[:, f'{a}'] = dt
        if d[f'{a}'] == 'box-cox':
            dt = power_transform(np.array(df.loc[:, f'{a}']).reshape((length, 1)),
                                 method='box-cox')
            df.loc[:, f'{a}'] = dt
        if d[f'{a}'] == 'yeo-johnson':
            dt = power_transform(np.array(df.loc[:, f'{a}']).reshape((length, 1)),
                                 method='yeo-johnson')
            df.loc[:, f'{a}'] = dt

    df.to_excel(f'{args.out_df}', header=True, index=False)


if __name__ == '__main__':
    main()
