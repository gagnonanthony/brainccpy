#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute K-Means clustering on a datasets.
"""


import argparse
import pandas as pd
from brainccpy.Clustering.utils import remove_nans, visualize_clustering
from brainccpy.Clustering.kmeans import (elbow_method,
                                         cluster_pipeline)
import matplotlib.pyplot as plt


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_df',
                   help='Input dataframe.')
    p.add_argument('--init_method', choices=['random', 'k-means++'],
                   help='Initialization method.')
    p.add_argument('--n_init', type=int,
                   help='Number of initialization.')
    p.add_argument('--max_iter', type=int,
                   help='Maximum iterations.')
    p.add_argument('--output_dir',
                   help='Output folder.')
    p.add_argument('--random_seed', required=False, default=1234,
                   help='Random initialization seed.')
    p.add_argument('--verbose', action='store_true', required=False,
                   help='If applied, verbose mode is activated.')

    tsfm = p.add_mutually_exclusive_group()
    tsfm.add_argument('--quantile', action='store_true',
                      help='Apply quantile transformation to data before clustering.')
    tsfm.add_argument('--scaling', action='store_true',
                      help='Apply scaling to data before clustering')

    p.add_argument('--out_dist', choices=['normal', 'uniform'],
                   help='Distribution to produce following transformation. (If quantile'
                        'transformation is selected.)')
    p.add_argument('--nb_quant', type=int, default=100,
                   help='Number of quantile to fit the data into (If quantile'
                        'transformation is selected.)')

    viz = p.add_mutually_exclusive_group()
    viz.add_argument('--pca', action='store_true',
                     help='Use PCA method to visualize clustering results.')
    viz.add_argument('--tsne', action='store_true',
                     help='Use TSNE method to visualize clustering results.')

    p.add_argument('--perplexity', required=False, default=30,
                   help='Perplexity value to use in TSNE algorithm (if selected).')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    random_seed = args.random_seed

    if args.verbose:
        verbose = 1
    else:
        verbose = 0

    df = pd.read_excel(args.in_df)
    df = remove_nans(df)
    clust = df.iloc[:, 1:len(df.columns)]
    length = len(clust.columns)

    if args.quantile:
        sse, elbow_plot, elbow = elbow_method(df=clust,
                                              cluster_limit=50,
                                              init=f'{args.init_method}',
                                              n_init=args.n_init,
                                              max_iter=args.max_iter,
                                              t_method='quant',
                                              nb_qt=args.nb_quant,
                                              output_dist=f'{args.out_dist}',
                                              random_state=random_seed,
                                              verbose=verbose)
    else:
        sse, elbow_plot, elbow = elbow_method(df=clust,
                                              cluster_limit=50,
                                              init=f'{args.init_method}',
                                              n_init=args.n_init,
                                              max_iter=args.max_iter,
                                              t_method='scaled',
                                              random_state=random_seed,
                                              verbose=verbose)

    plt.text(x=len(sse)/2, y=max(sse)/2, s=f'Optimal number \n of clusters : {elbow}')
    plt.savefig(f'{args.output_dir}/elbow_graph.png')
    plt.cla()
    plt.clf()

    if args.quantile:
        pipe_final = cluster_pipeline(df=clust,
                                      output_dist=f'{args.out_dist}',
                                      n_clusters=elbow,
                                      init_method=f'{args.init_method}',
                                      nb_init=args.n_init,
                                      max_iter=args.max_iter,
                                      t_method='quant',
                                      nb_qt=args.nb_quant,
                                      random_state=random_seed,
                                      verbose=verbose)
        data_final = pipe_final['Transform'].transform(clust)
    else:
        pipe_final = cluster_pipeline(df=clust,
                                      output_dist=f'{args.out_dist}',
                                      n_clusters=elbow,
                                      init_method=f'{args.init_method}',
                                      nb_init=args.n_init,
                                      max_iter=args.max_iter,
                                      t_method='scaled',
                                      nb_qt=args.nb_quant,
                                      random_state=random_seed,
                                      verbose=verbose)
        data_final = pipe_final['Scaling'].transform(clust)

    labels_final = pipe_final['kmeans'].labels_

    data_final = pd.DataFrame(data_final, columns=clust.columns)

    data_final.insert(len(data_final.columns), 'Cluster', labels_final)

    # Visualizing clustering results.
    if args.pca:
        visualize_clustering(data_final, f'{args.output_dir}/',
                             method='PCA')
    else:
        visualize_clustering(data_final, f'{args.output_dir}/',
                             method='TSNE',
                             perplexity=args.perplexity)

    data_final.to_excel(f'{args.output_dir}/clustering_data.xlsx', header=True, index=False)


if __name__ == '__main__':
    main()
