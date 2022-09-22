# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
    :param df:
    :return:
    """
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
