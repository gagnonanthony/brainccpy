# -*- coding: utf-8 -*-

import seaborn as sns
import numpy as np
from sklearn.preprocessing import (QuantileTransformer,
                                   PowerTransformer,
                                   StandardScaler)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation


def plot_dist(df, ind1, ind2):
    """
    Script to plot distribution and correlation map of a subset of a dataframe.
    :param df:      Pandas dataframe.
    :param ind1:    Indice of first column in the interval
    :param ind2:    Indice of the second column in the interval
    :return:        Seaborn object (graph).
    """
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    viz = df.iloc[:, ind1:(ind2 + 1)]
    g = sns.PairGrid(viz)
    g.map_upper(sns.histplot)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.histplot, kde=True)

    return g


def remove_nans(df):
    """
    Script design to remove all rows containing NaNs.
    :param df:  Pandas dataframe object.
    :return:    Pandas dataframe object.
    """
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def quantile_transform(df, nb_qt, output_dist):
    """

    :param df:
    :param nb_qt:
    :return:
    """
    qt = QuantileTransformer(n_quantiles=nb_qt, random_state=0,
                             output_distribution=f'{output_dist}')
    out = qt.fit_transform(df)

    return out


def power_transform(df, method):
    """

    :param df:
    :param method:
    :return:
    """
    pw = PowerTransformer(method=f'{method}', standardize=True)
    out = pw.fit_transform(df)

    return out


def standard_scale(df):
    """

    :param df:
    :return:
    """
    stsc = StandardScaler()
    out = stsc.fit_transform(df)

    return out


def visualize_clustering(df, output, method='PCA', perplexity=30):
    """

    :param df:
    :param method:
    :param perplexity:
    :param output:
    :return:
    """

    if method == 'PCA':
        m_1d = PCA(n_components=1)
        m_2d = PCA(n_components=2)
        m_3d = PCA(n_components=3)
    elif method == 'TSNE':
        m_1d = TSNE(n_components=1, perplexity=perplexity, learning_rate='auto',
                    n_iter=2000, metric='euclidean', init='pca', verbose=1)
        m_2d = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto',
                    n_iter=2000, metric='euclidean', init='pca', verbose=1)
        m_3d = TSNE(n_components=3, perplexity=perplexity, learning_rate='auto',
                    n_iter=2000, metric='euclidean', init='pca', verbose=1)

    PCs_1d = pd.DataFrame(m_1d.fit_transform(df.drop(["Cluster"], axis=1)))
    PCs_1d.columns = ["PC1_1d"]
    PCs_2d = pd.DataFrame(m_2d.fit_transform(df.drop(["Cluster"], axis=1)))
    PCs_2d.columns = ["PC1_2d", "PC2_2d"]
    PCs_3d = pd.DataFrame(m_3d.fit_transform(df.drop(["Cluster"], axis=1)))
    PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]

    PC_df = pd.concat([PCs_1d, PCs_2d, PCs_3d, df['Cluster']], axis=1, join='inner')
    PC_df['dummy'] = 0

    # Plotting results.
    sns.countplot(data=PC_df, x='Cluster', palette='Spectral',
                  ).set(title='Number of samples per clusters.',
                        xlabel='Clusters', ylabel='Nb of samples')
    plt.savefig(f'{output}/count_plot.pdf', type='pdf')
    plt.cla()
    plt.clf()

    sns.scatterplot(data=PC_df,
                    x='PC1_1d',
                    y='dummy',
                    hue='Cluster',
                    legend='full',
                    palette='Spectral',
                    ).set(title='1D representation of clustering algorithm.',
                          xlabel='PC1', ylabel='')
    plt.savefig(f'{output}/1d_results.pdf', type='pdf')
    plt.cla()
    plt.clf()

    sns.scatterplot(data=PC_df,
                    x='PC1_2d',
                    y='PC2_2d',
                    hue='Cluster',
                    legend='full',
                    palette='Spectral',
                    ).set(title='2D representation of clustering algorithm.',
                          xlabel='PC1', ylabel='PC2')
    plt.savefig(f'{output}/2d_results.pdf', type='pdf')
    plt.cla()
    plt.clf()

    sns.set(style='darkgrid')
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    cmap = ListedColormap(sns.color_palette('Spectral', max(PC_df['Cluster'])+1).as_hex())
    colors = sns.color_palette('Spectral', max(PC_df['Cluster'])+1)

    list_lab = list(range(0, max(PC_df['Cluster'])+1))

    custom_legend = [Line2D([], [], marker='.', color=colors[i], markersize=15,
                            linestyle=None, linewidth=None) for i in list_lab]


    x = PC_df['PC1_3d']
    y = PC_df['PC2_3d']
    z = PC_df['PC3_3d']

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D representation of clustering algorithm.')

    sc = ax.scatter(x, y, z, s=30, c=PC_df['Cluster'], marker='o', cmap=cmap, alpha=1)
    legend = ax.legend(handles=custom_legend, labels=[f'Cluster {i+1}' for i in list_lab],
                       loc='upper right', bbox_to_anchor=(1.05, 1), prop={'size': 8})

    def rotate(angle):
        ax.view_init(azim=angle)

    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2),
                                            interval=100)
    rot_animation.save(f'{output}/3d_results.gif', dpi=80, writer='imagemagick')




