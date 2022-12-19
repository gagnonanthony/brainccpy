# -*- coding: utf-8 -*-

import argparse
import logging
import itertools
import numpy as np
import shutil
import os


def add_overwrite_arg(parser):
    parser.add_argument(
        '-f', dest='overwrite', action='store_true',
        help='Force overwriting of the output files.')


def add_verbose_arg(parser):
    parser.add_argument('-v', action='store_true', dest='verbose',
                        help='If set, produces verbose output.')


def validate_input(parser, required, optional=None):
    """Function to validate the existence of the input.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    required: string or list of paths
        Required paths to be checked
    optional: string or list of paths
        Optional paths to be checked
    """
    def validate(path):
        if not os.path.isfile(path):
            parser.error('Input file {} does not exist.'.format(path))

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        validate(required_file)
    for optional_file in optional or []:
        if optional_file is not None:
            validate(optional_file)


def validate_output(parser, args, required, optional=None,
                    validate_dir=True):
    """
    Validate that output are available and doesn't exist. If so,
    validate that -f flag is true.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    args: argparse namespace
        Argument list.
    required: string or list of paths to files
        Required paths to be checked.
    optional: string or list of paths to files
        Optional paths to be checked.
    validate_dir: bool
        Test if output directory exists.
    """
    def validate(path):
        if os.path.isfile(path) and not args.overwrite:
            parser.error('Output file {} exists. Use -f flag'
                         'to overwrite.'.format(path))

        if validate_dir:
            path_dir = os.path.dirname(path)
            if path_dir and not os.path.isdir(path_dir):
                parser.error('Directory {} \n for a given output file'
                             'does not exists.'.format(path_dir))

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        validate(required_file)
    for optional_file in optional or []:
        if optional_file is not None:
            validate(optional_file)


def validate_output_dir(parser, args, required,
                        optional=None, create_dir=True):
    """
    Validate that all output directories exist.
    If not, create it.
    If exists and not empty, use -f flag.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    args: argparse namespace
        Argument list.
    required: string or list of paths to files
        Required paths to be checked.
    optional: string or list of paths to files
        Optional paths to be checked.
    create_dir: bool
        If true, create the directory if it does not exist.
    """
    def validate(path):
        if not os.path.isdir(path):
            if not create_dir:
                parser.error("Output directory {} doesn't exist.".format(path))
            else:
                os.makedirs(path, exist_ok=True)
        if os.listdir(path):
            if not args.overwrite:
                parser.error("Output directory {} isn't empty and some files could be"
                             "overwritten or even deleted. Use -f flag to overwrite.".format(path))
            else:
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    try:
                        if os.path.isfile(file):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)

    if isinstance(required, str):
        required = [required]
    if isinstance(optional, str):
        optional = [optional]

    for cur_dir in required:
        validate(cur_dir)
    for opt_dir in optional or []:
        if opt_dir:
            validate(opt_dir)


def compute_matrices_density(mat):
    """
    Function to compute the density of binary matrices
    :param mat:     Binary matrices (.npy)
    :return:        Density values (in %)
    """
    mat = np.load(mat)
    unique, counts = np.unique(mat, return_counts=True)
    dic = dict(zip(unique, counts))

    # Compute Density
    tot = mat.shape[0] * mat.shape[1]
    dens = dic[1] / tot * 100

    return dens
