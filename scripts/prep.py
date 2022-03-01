import argparse
import json

from evergraph.utils.logger_utils import setup_logger
from evergraph.prep.data_prepper import DataPrepper

def parse_arguments():
    parser = argparse.ArgumentParser(
            description = "Reorganize and preprocess outputs from HiggsDNA for training with EverGraphDNN")

    # Required arguments
    parser.add_argument(
            "--input_dir",
            required=True,
            type=str,
            help="path to input directory with `merged_nominal.parquet` and `summary.json` from HiggsDNA")

    parser.add_argument(
            "--output_dir",
            required=True,
            type=str,
            help="path to output directory")

    # Optional arguments
    parser.add_argument(
            "--log-level",
            required=False,
            default="DEBUG",
            type=str,
            help="Level of information printed by the logger") 

    parser.add_argument(
            "--log-file",
            required=False,
            type=str,
            help="Name of the log file")

    parser.add_argument(
            "--short",
            required=False,
            action="store_true",
            help="flag to just process 10k events")

    parser.add_argument(
            "--selection",
            required=False,
            default=None,
            type=str,
            help="selection to apply on events")

    parser.add_argument(
            "--objects",
            required=False,
            default=None,
            type=str,
            help="csv list of objects to process (photons, leptons, jets, met)")

    return parser.parse_args()


def main(args):
    logger = setup_logger(args.log_level, args.log_file)
    
    data_prepper = DataPrepper(**vars(args))
    data_prepper.run()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
