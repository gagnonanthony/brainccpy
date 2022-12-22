# -*- coding: utf-8 -*-
import logging

import numpy as np
import logging


def track_clustering(mat):
    """
    Function to classify connections in clusters (defined as connections linking
    common regions).
    :param mat:         Binary matrix containing connections to sort. (.npy)
    :return:            Dictionary of clusters.
    """

    xindex = mat.shape[0]
    yindex = mat.shape[1]

    mat = np.triu(mat)

    if xindex == yindex:
        logging.info('Input matrix is symmetrical')
    else:
        logging.info('Input matrix is asymmetrical.')

    # Count number of non-zero connections in the matrix.
    unique, counts = np.unique(mat, return_counts=True)
    dic = dict(zip(unique, counts))
    logging.info('Matrix contains {} connections. Extracting connections ...'.format(dic[1]))

    # Extracting all connections with non-zero values in a table.
    x = []
    y = []
    for i in range(0, xindex):
        for j in range(0, yindex):
            if mat[i, j] == 1:
                x.append(i+1)
                y.append(j+1)
    pairs = np.rollaxis(np.array((x, y), dtype=int), 1)
    logging.info('Lookup table created. Clustering connections...')

    # Clustering connections.
    cluster_dict = {}
    n = 1
    while len(pairs) > 0:
        i = np.array(pairs[0][0])
        k = 1
        while k <= len(pairs):
            # Sorting clusters
            ix = pairs[np.isin(pairs[:, 0], i), 1]
            iy = pairs[np.isin(pairs[:, 1], i), 0]
            im = np.append(ix, iy)
            iu = np.unique(np.append(i, im))
            k += 1
            if np.array_equal(iu, i):
                break
            else:
                i = iu
        logging.info(f'Cluster #{n} took {k} iterations to extract.')
        # Saving cluster in dictionary as pairs (X_Y).
        x = pairs[np.isin(pairs[:, 1], i), 0]
        y = pairs[np.isin(pairs[:, 0], i), 1]
        assert len(x) == len(y), "Mismatch in number of starting and receiving regions."
        cluster_dict[f'Cluster_{n}'] = [f'{x[c]}_{y[c]}' for c in range(0, len(x))]
        pairs = np.extract(~np.isin(pairs, i), pairs)
        pairs = pairs.reshape(int(len(pairs)/2), 2)
        n += 1
    logging.info(f'{n-1} clusters extracted. No remaining connections.')

    return cluster_dict


