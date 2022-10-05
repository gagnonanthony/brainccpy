# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from brainccpy.Clustering.utils import QuantileTransformer


def elbow_method(df, cluster_limit, init='k-means++', n_init=20, max_iter=1000, t_method='quant',
                 nb_qt=100, output_dist='normal', random_state=1234, verbose=0):
    """
    Function perform the elbow method for the optimal number of cluster to use.
    :param df:                  Pandas dataframe.
    :param cluster_limit:       Higher number of cluster to evaluate.
                                Algorithm will go from k=1 to k=cluster_limit.
    :param init:                Initiation state. ['random' or 'k-means++']
    :param n_init:              Number of initializations to perform. Algorithm
                                returns the initializations with the best SSE.
    :param max_iter:            Number of max iterations to perform.
    :param t_method:            Method for transforming data ('quant' or 'scaled')
    :param nb_qt:               Number of quantile to use if 'quant' is selected.
    :param output_dist:         Outputted distribution following 'quant' transform.
    :param random_state:
    :param verbose:
    :return:
    """

    kmeans_kwargs = {
        'init_method': f'{init}',
        'nb_init': n_init,
        'max_iter': max_iter,
        't_method': f'{t_method}',
        'nb_qt': nb_qt,
        'output_dist': f'{output_dist}',
        'random_state': random_state,
        'verbose': verbose,
    }

    sse = []
    for k in range(1, cluster_limit):
        pipe = cluster_pipeline(df, k, **kmeans_kwargs)
        sse.append(pipe['kmeans'].inertia_)

    # Plotting the results.
    plot = plt.plot(list(range(1, cluster_limit)), sse)
    plt.xticks(list(range(1, cluster_limit)))
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')

    # Using the Kneed package.
    kl = KneeLocator(
        range(1, cluster_limit), sse, curve='convex', direction='decreasing'
    )
    elbow = kl.elbow

    return sse, plot, elbow


def silhouette_coef(df, cluster_limit, init='k-means++', n_init=20, max_iter=1000, t_method='quant',
                 nb_qt=100, output_dist='normal', random_state=1234):
    """

    :param df:
    :param cluster_limit:
    :param init:
    :param n_init:
    :param max_iter:
    :param t_method:
    :param nb_qt:
    :param output_dist:
    :param random_state:
    :return:
    """

    kmeans_kwargs = {
        'init_method': f'{init}',
        'nb_init': n_init,
        'max_iter': max_iter,
        't_method': f'{t_method}',
        'nb_qt': nb_qt,
        'output_dist': f'{output_dist}',
        'random_state': random_state,
    }

    silhouette_coefficients = []

    for k in range(2, cluster_limit):
        pipe = cluster_pipeline(df, k, **kmeans_kwargs)
        score = silhouette_score(df, pipe['kmeans'].labels_)
        silhouette_coefficients.append(score)

    silhouette_plot = plt.plot(range(2, cluster_limit), silhouette_coefficients)
    plt.xticks(range(2, cluster_limit))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient')

    return silhouette_plot, silhouette_coefficients


def cluster_pipeline(df, n_clusters, init_method='k-means++', nb_init=10, max_iter=1000,
                     t_method='quant', nb_qt=100, output_dist='normal', random_state=1234,
                     verbose=0):
    """

    :param df:
    :param n_clusters:
    :param init_method:
    :param nb_init:
    :param max_iter:
    :param t_method:
    :param nb_qt:
    :param output_dist:
    :param random_state:
    :param verbose:
    :return:
    """

    transf_kwargs = {
        'n_quantiles': nb_qt,
        'output_distribution': f'{output_dist}'
    }

    kmeans_kwargs = {
        'n_clusters': n_clusters,
        'init': f'{init_method}',
        'n_init': nb_init,
        'max_iter': max_iter,
        'random_state': random_state,
        'verbose': verbose,
    }

    if t_method == 'quant':
        pipe = Pipeline(
            [
                ('Transform', QuantileTransformer(
                    **transf_kwargs,
                )),
                ('kmeans', KMeans(
                    **kmeans_kwargs,
                ),),
            ],
            verbose=True
        )

    else:
        pipe = Pipeline(
            [
                ('Scaling', StandardScaler()),
                ('kmeans', KMeans(
                    **kmeans_kwargs,
                ),),
            ],
            verbose=True
        )

    pipe.fit(df)

    return pipe
