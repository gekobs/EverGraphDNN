import awkward

import logging
logger = logging.getLogger(__name__)

class DataPrepper():
    """
    Class to read flat .parquet data (expected to be an output of HiggsDNA)
    and prep:
        - reorganize existing fields into training features and labels
        - preprocess features and labels
        - up/down sample classes

    See HiggsDNA project : https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA

    :param input_dir: path to input directory with `merged_nominal.parquet` and `summary.json` from HiggsDNA
    :type input_dir: str
    :param output_dir: path to output directory
    :type output_dir: str
    :param short: flag to just process the first 1000 events
    :type short: bool
    """
    def __init__(self, input_dir, output_dir, short = False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.short = short


