#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to extract metrics for individual pair of ROIs from a connectivity matrix.
Input directory should be either a connectoflow output (in that case, use --connectoflow)
or should respect this structure : ${input}/${subject1}.npy
                                           /${subject2}.npy
                                           /...
"""

import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--input', nargs='+', required=False,
                   help='Path to the input directory folder. Not necessary if the input \n'
                        'folder is a connectoflow output structure.')

    p.add_argument('--output', required=False, default='./',
                   help='Output filename (.csv).')

    roi = p.add_argument_group(title='Specifying connections to extract.',
                               description='Two options available : all connections or \n'
                                           'the one specified by a binary mask.')
    roi.add_argument('--in_mask', required=False,
                     help='Path to the binary matrix (.npy) selecting connections of interest to extract metrics.')
    roi.add_argument('--all', required=False, default=False, action='store_true',
                     help='If true, will extract all measures from all connections.')

    conn = p.add_argument_group(title='Connectoflow options',
                                description='Options if matrices to extract are inside \n'
                                            'a connectoflow output structure.')
    conn.add_argument('--connectoflow', action='store_true', required=False,
                      help='If true, will assume that the directory is a connectoflow output.')
    conn.add_argument('--connectoflow_folder', required=False,
                      help='If --connectoflow is selected, please provide the connectoflow folder.')
    conn.add_argument('--in_ID_list', required=False,
                      help='Path of the subject ID list in a .txt file.')
    conn.add_argument('--in_metrics', required=False,
                      help='Abbreviation of the metric to extract (ex : ad, afd, md, etc.).')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.all:
        if args.connectoflow:
            subjects = open(args.in_ID_list).read().split()
            mat = np.load(f'{args.connectoflow_folder}/{subjects[0]}/Compute_Connectivity/{args.in_metrics}.npy')
            mask = np.ones([mat.shape[0], mat.shape[1]])
        else:
            mat = np.load(f'{args.input[0]}')
            mask = np.ones([mat.shape[0], mat.shape[1]])
    else:
        mask = np.load(args.in_mask)

    labels = pd.DataFrame(mask)
    labels_col = labels.columns
    labels_row = labels.index
    labels_value = labels.values
    labels_value = np.triu(labels_value)

    columns = []
    results = []

    for i in range(labels_col.shape[0]):
        for j in range(labels_row.shape[0]):
            cond = labels_value[i, j]
            if cond == 1:
                columns.append(str(int(labels_col[i] + 1)) + '_' + str(int(labels_row[j] + 1)))

    if args.connectoflow:
        for subject in tqdm(subjects):
            metrics = np.triu(np.load(f'{args.directory}/{subject}/Compute_Connectivity/{args.in_metrics}.npy'))
            output = []
            for i in range(labels_col.shape[0]):
                for j in range(labels_row.shape[0]):
                    cond = labels_value[i, j]
                    if cond == 1:
                        out = metrics[i, j]
                        output.append(out)
            res = [x for x in output]
            results.append(res)
    else:
        for i in tqdm(args.input):
            metrics = np.triu(np.load(f'{i}'))
            output = []
            for i in range(labels_col.shape[0]):
                for j in range(labels_row.shape[0]):
                    cond = labels_value[i, j]
                    if cond == 1:
                        out = metrics[i, j]
                        output.append(out)
            res = [x for x in output]
            results.append(res)

    final = pd.DataFrame(results, columns=columns)

    if args.connectoflow:
        final.index = subjects
        final.to_csv(f'{args.output}', header=True, index=subjects, index_label='IDs')
    else:
        final.index = args.input
        final.to_csv(f'{args.output}', header=True, index=args.input, index_label='IDs')


if __name__ == "__main__":
    main()
