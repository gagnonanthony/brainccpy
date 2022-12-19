#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute various mathematical operation on a numpy matrices.
"""

import argparse
import logging
from brainccpy.io.utils import (validate_input,
                                compute_matrices_density,
                                validate_output,
                                validate_output_dir,
                                add_verbose_arg,
                                add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--input',
                   help='Path to the output folder.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    density = compute_matrices_density(args.input)
    print(density)


if __name__ == '__main__':
    main()
