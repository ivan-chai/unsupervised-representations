#!/usr/bin/env python3
"""Compute maximum mutual information for given embedding size and centroid sigma."""
import argparse

from urep.estimators.gcpc import get_maximal_mutual_information


def parse_arguments():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dim", help="Embedding size", type=int, required=True)
    parser.add_argument("--centroid-sigma2", help="Centroid variance", type=float, required=True)
    return parser.parse_args()


def main(args):
    print(get_maximal_mutual_information(args.dim, args.centroid_sigma2))


if __name__ == "__main__":
    main(parse_arguments())
