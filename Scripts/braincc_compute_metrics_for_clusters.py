#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import logging
import os

import json
import numpy as np
from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             load_matrix_in_any_format)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--cluster_json', required=True,
                   help='Json file containing clusters. Output from \n'
                        'braincc_connections_clustering.py')
    p.add_argument('--list_id', required=True,
                   help='.txt file containing the list of id to extract.')
    p.add_argument('--conn_dir', required=True,
                   help='Directory containing the connectoflow output. \n'
                        'Needed to fetch the selected connectivity matrices.')
    p.add_argument('--metrics', required=True,
                   help='Metrics to extract from the connectoflow output.')
    p.add_argument('--output', required=True,
                   help='Filename for the outputted excel sheet (.xlsx)')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    assert_inputs_exist(parser, args.cluster_json+args.list_id+args.conn_dir+args.metrics)
    assert_outputs_exist(parser, args, args.output)

    with open(f'{args.cluster_json}', 'r') as f:
        cluster_dict = json.load(f)

    subjects = open(args.list_id).read().split()

    for metric in args.metrics:

        for sub in subjects:
            mat = load_matrix_in_any_format(f'{args.conn_dir}/{sub}/Compute_Connectivity/{metric}.npy')
            shape = mat.shape
            for i in cluster_dict.keys():
                mask = np.empty((shape))
                edges = cluster_dict[f'{i}']
                for i in edges:
                    x, y = i.split('_')
                    mask[int(x)-1, int(y)-1] = 1
                # Keep only values over 0
                non_zero = mask > 0
                mean = np.mean(mat * mask, where=non_zero)

