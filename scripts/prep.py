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

    return parser.parse_args()


def main(args):
    logger = setup_logger(args.log_level, args.log_file)
    
    data_prepper = DataPrepper(**vars(args))
    data_prepper.run()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
