import json
import xgboost
import awkward
import pandas
import os
import math
import glob
import matplotlib.pyplot as plt
from evergraph.prep.data_prepper import DUMMY_VALUE
import numpy as np

from sklearn import metrics

import logging
logger = logging.getLogger(__name__)

import evergraph.utils.bdt_utils as bdt_utils
from evergraph.utils.misc_utils import update_dict
from evergraph.algorithms.losses import selective_mse
from evergraph.algorithms.generators import EvergraphGenerator


DEFAULT_OPTIONS = {
    "signal": [
        "ttHH_ggbb", 
    ],
    "background": [
        "ttH_M125",
    ],
    "training_features": [
   
    ],
    "targets" : [
        #"target_has_HggHiggs",
        "target_has_HbbHiggs",
        #"target_HggHiggs_pt", "target_HggHiggs_eta",
        #"target_has_HbbHiggs", "target_HbbHiggs_pt", "target_HbbHiggs_eta",
        #"target_has_HggHiggs", "target_HggHiggs_pt", "target_HggHiggs_eta", "target_HggHiggs_phi", "target_HggHiggs_mass",
        #"target_has_HbbHiggs", "target_HbbHiggs_pt", "target_HbbHiggs_eta", "target_HbbHiggs_phi", "target_HbbHiggs_mass",
        #"target_has_HttHiggs", "target_HttHiggs_pt", "target_HttHiggs_eta", "target_HttHiggs_phi", "target_HttHiggs_mass",
        #"target_has_HwwHiggs", "target_HwwHiggs_pt", "target_HwwHiggs_eta", "target_HwwHiggs_phi", "target_HwwHiggs_mass",
        #"target_has_Top_1", "target_Top_1_pt", "target_Top_1_eta", "target_Top_1_phi", "target_Top_1_mass",
        #"target_has_Top_2", "target_Top_2_pt", "target_Top_2_eta", "target_Top_2_phi", "target_Top_2_mass",
    ],
    "mva": {
        "param": {
            "max_depth": 4,
            "eta": 0.2,
            "objective": "binary:logistic",
            "eval_metric": "error",
            "nthread": 12
        },
        "n_trees": 500,
        "early_stopping": True,
        "early_stopping_rounds": 5
    }
}


class BDTHelper():
    """
    Class to read events in from an parquet file and perform necessary
    preprocessing, sample labeling, etc and perform customisable BDT training 
    """

    def __init__(self, input_dir, output_dir, config = {}, **kwargs):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.files = glob.glob(self.input_dir + "/*.parquet")
        df = awkward.concatenate([awkward.from_parquet(f) for f in self.files] )
        sorted(self.files)
        self.file_map={}
        self.file_map["train"] = self.files[0::2]
        self.file_map["test"] = self.files[1::2] 
        self.data_array = {}
        for split in ["train","test"]:
            self.data_array[split] = awkward.concatenate(
                [awkward.from_parquet(f) for f in self.file_map[split] ] 
            ) 

        logger.debug("[BDTHelper] Running on a total of %s events. Training/Testing splits %s/%s"%(len(df),len(self.data_array["train"]),len(self.data_array["test"]) ) )
        
        self.config = self.config = update_dict(original = DEFAULT_OPTIONS, new = config)
        self.output_tag = kwargs.get("output_tag", "")
        self.made_dmatrix = False

    def train(self):
        self.make_dmatrix()
    
        eval_list = [(self.events["train"]["dmatrix"], "train"), (self.events["test"]["dmatrix"], "test")]
        progress = {}

        logger.debug("[BDTHelper] Training BDT with options:")
        logger.debug(self.config["mva"]["param"])

        if self.config["mva"]["early_stopping"]:
            n_early_stopping =  self.config["mva"]["early_stopping_rounds"]
            logger.debug("[BDTHelper] early stopping with %d rounds (%d maximum)" % (n_early_stopping, self.config["mva"]["n_trees"]))
        else:
            logger.debug("[BDTHelper] using %d trees (no early stopping)" % (self.config["mva"]["n_trees"]))
            n_early_stopping = None
        
        self.bdt = xgboost.train(
            self.config["mva"]["param"],
            self.events["train"]["dmatrix"],
            self.config["mva"]["n_trees"],
            eval_list, evals_result = progress,
            early_stopping_rounds = n_early_stopping
        )

        return self.bdt

    def make_dmatrix(self):
        for split in self.events.keys():
            self.events[split]["dmatrix"] = xgboost.DMatrix(
                self.events[split]["X"],
                self.events[split]["y"],
            )
        self.made_dmatrix = True
        return

    def predict_from_df(self, df):
        X = xgboost.DMatrix(df["objects"])
        return self.bdt.predict(X)

    def predict(self):
        if not self.made_dmatrix:
            self.make_dmatrix()
        self.prediction = {}
        for split in self.events.keys():
            self.prediction[split] = self.bdt.predict(self.events[split]["dmatrix"])
        return self.prediction

    def save_weights(self):
        self.weights_file =  "output/" + self.output_tag + ".xgb"
        self.summary_file = self.weights_file.replace(".xgb", ".json")
        self.bdt.save_model(self.weights_file)
        summary = {
            "config" : self.config,
            "weights" : self.weights_file
        }
        with open(self.summary_file, "w") as f_out:
            json.dump(summary, f_out, sort_keys = True, indent = 4)

        return summary

    def load_weights(self, weight_file):
        self.bdt = xgboost.Booster()
        self.bdt.load_model(weight_file)

    def load_events(self):
        self.events = {}
        for split in ["train", "test"]:
            self.events[split] = {}
            self.data_array[split]["flat_objects"] = awkward.flatten(self.data_array[split]["objects"],axis=-1)
            self.data_array[split]["flat_objects"] = awkward.nan_to_num(self.data_array[split]["flat_objects"],
                                                        nan=DUMMY_VALUE, posinf=DUMMY_VALUE, neginf=DUMMY_VALUE)
            features = awkward.to_numpy(self.data_array[split]["flat_objects"])
            labels = awkward.to_pandas(self.data_array[split]["target_has_HbbHiggs"]==1)
            self.events[split]["X"] = features
            self.events[split]["y"] = labels
        
        return
        
    def evaluate_performance(self):
        self.performance = {}
        for split in self.events.keys():
            self.performance[split] = bdt_utils.calc_roc_and_unc(
                self.events[split]["y"],
                self.prediction[split],
                # self.events[split],
                n_bootstrap = 25
            )

            logger.debug("[MVA_HELPER] Performance (%s set): AUC = %.3f +/- %.3f" % (split, self.performance[split]["auc"], self.performance[split]["auc_unc"]))
        
        self.make_plots()
        self.save_performance()
        
    def make_plots(self):
        self.plots = []
        for split in self.events.keys():
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.yaxis.set_ticks_position('both')
            ax1.grid(True)

            ax1.plot(self.performance[split]["fpr"],
                     self.performance[split]["tpr"],
                     color = "red",
                     label = "BDT AUC: %.3f +/- %.3f" % (self.performance[split]["auc"], self.performance[split]["auc_unc"]))
            ax1.fill_between(self.performance[split]["fpr"],
                             self.performance[split]["tpr"] - (self.performance[split]["tpr_unc"]/2.),
                             self.performance[split]["tpr"] + (self.performance[split]["tpr_unc"]/2.),
                             color = "red",
                             alpha = 0.25, label = r'$\pm 1\sigma')

            plt.xlim([-0.05,1.05])
            plt.ylim([-0.05,1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc = "lower right")
            plot_name = "output/roc_comparison_%s_%s.pdf" % (self.output_tag, split)
            plt.savefig(plot_name)
            self.plots.append(plot_name)
            plt.clf()

    def save_performance(self):
        """
        Save roc curves to npz file
        """
        self.npz_file = "output/" + self.output_tag + ".npz"
        self.npz_results = {}
        for split in self.events.keys():
            for metric in self.performance[split].keys():
                self.npz_results[metric + "_" + split] = self.performance[split][metric]
            for info in ["y"]:
                self.npz_results[info + "_" + split] = self.events[split][info]
        np.savez(self.npz_file, **self.npz_results)        
        return

    def run(self):
        self.load_events()
        self.train()
        self.predict()
        self.evaluate_performance()
        self.save_weights()