#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os

import numpy as np
from brainccpy.viz.utils import track_clustering
from brainccpy.io.utils import (add_verbose_arg,
                                add_overwrite_arg,
                                validate_input,
                                validate_output_dir)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--input',
                   help='Input binary matrix containing significant connections to cluster.')
    p.add_argument('--run_decompose', action='store_true', default=True,
                   help="If True, script will run scil_save_connections_from_hdf5.py to \n"
                        "save raw connections.")
    p.add_argument('--hdf5', required=False,
                   help='HDF5 filename (.h5) for a single subject containing decomposed connections.')
    p.add_argument('--in_connections', required=False,
                   help='Folder containing the already saved .trk files for all connections. \n'
                        'Connections should be saved in the format X_Y.trk in order to be correctly \n'
                        'read by scil_streamlines_math.py.')
    p.add_argument('--output',
                   help='Directory output folder.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    validate_input(parser, args.input, args.hdf5)
    validate_output_dir(parser, args, args.output)

    # Create clusters.
    mat = np.load(args.input)
    cluster_dict = track_clustering(mat, verbose=(True if args.verbose else False))

    if args.run_decompose:
        if args.hdf5 is None:
            parser.error('To use --run_decompose, you need to provide a hdf5 file'
                         '(--hdf5).')

        # Creating output directory.
        out_dir_raw = os.path.join(args.output, 'Raw_Connections')
        os.mkdir(out_dir_raw)

        # Flattening the connections' list.
        flat_conn = [conn for cluster in list(cluster_dict.values()) for conn in cluster]

        # Run scil_save_connections_from_hdf5.py
        os.system(f'scil_save_connections_from_hdf5.py {args.hdf5} {out_dir_raw} --edge_keys {" ".join(flat_conn)} -f')
    else:
        out_dir_raw = args.in_connections

    # Merge individuals connections into one cluster files.
    out_dir = os.path.join(args.output, 'Clusters')
    os.mkdir(out_dir)

    for cluster in cluster_dict.keys():
        files = [f'{out_dir_raw}/{f}.trk' for f in cluster_dict[f'{cluster}']]

        os.system(f'scil_streamlines_math.py concatenate {" ".join(files)} {out_dir}/{cluster}.trk --robust -f -v')


if __name__ == '__main__':
    main()
