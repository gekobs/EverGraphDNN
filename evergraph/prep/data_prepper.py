import awkward
import numpy
import json
import os

import logging
logger = logging.getLogger(__name__)

DUMMY_VALUE = -999.

DEFAULT_OPTIONS = {
        "objects" : {
            "n_jets" : 8,
            "n_leptons" : 4,
            "n_photons" : 2
        }
}

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
    :param short: flag to just process 10k events
    :type short: bool
    """
    def __init__(self, input_dir, output_dir, short = False, **kwargs):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.short = short
        self.options = DEFAULT_OPTIONS


    def run(self):
        """

        """
        logger.debug("[DataPrepper : run] Running data prepper.")
        events = self.load_data()
        self.set_objects(events)
        self.set_targets(events)
        self.write_data()


    def load_data(self):
        """

        """
        self.f_events = self.input_dir + "/merged_nominal.parquet"
        self.f_summary = self.input_dir + "/summary.json"

        with open(self.f_summary, "r") as f_in:
            self.process_map = json.load(f_in)["sample_id_map"]
        
        events = awkward.from_parquet(self.f_events)

        self.n_events = { "total" : len(events) }
        logger.debug("[DataPrepper : load_data] Loaded %d events from file '%s' and summary json '%s'" % (self.n_events["total"], self.f_events, self.f_summary))

        if self.short:
            idx = numpy.random.randint(low = 0, high = self.n_events["total"], size = 100000)
            events = events[idx]
            self.n_events["total"] = len(events)

        for proc, proc_id in self.process_map.items():
            events_proc = events[events.process_id == proc_id]
            self.n_events[proc] = len(events_proc)

            logger.debug("\t %s : %d events (%.1f percent)" % (proc, self.n_events[proc], 100. * float(self.n_events[proc]) / float(self.n_events["total"])))

        return events


    def set_objects(self, events):
        """

        """
        self.n_objects = self.options["objects"]["n_jets"] + self.options["objects"]["n_leptons"] + self.options["objects"]["n_photons"]
        for i in range(self.n_objects + 1):
            events["object_%d_pt" % i] = awkward.zeros_like(events.process_id, dtype = numpy.float32)
            events["object_%d_eta" % i] = awkward.zeros_like(events.process_id, dtype = numpy.float32)
            events["object_%d_phi" % i] = awkward.zeros_like(events.process_id, dtype = numpy.float32)
            events["object_%d_mass" % i] = awkward.zeros_like(events.process_id, dtype = numpy.float32)
            events["object_%d_charge" % i] = awkward.zeros_like(events.process_id, dtype = numpy.int8)
            events["object_%d_is_jet" % i] = awkward.zeros_like(events.process_id, dtype = numpy.bool)
            events["object_%d_is_pho" % i] = awkward.zeros_like(events.process_id, dtype = numpy.bool)
            events["object_%d_is_ele" % i] = awkward.zeros_like(events.process_id, dtype = numpy.bool)
            events["object_%d_is_muo" % i] = awkward.zeros_like(events.process_id, dtype = numpy.bool)
            events["object_%d_is_tau" % i] = awkward.zeros_like(events.process_id, dtype = numpy.bool)
            events["object_%d_is_met" % i] = awkward.zeros_like(events.process_id, dtype = numpy.bool)
            events["object_%d_btag" % i] = awkward.zeros_like(events.process_id, dtype = numpy.float32)

        # 1. Set photons 
        events["object_0_pt"] = numpy.log(events.LeadPhoton_pt) 
        events["object_0_eta"] = events.LeadPhoton_eta
        events["object_0_phi"] = events.LeadPhoton_phi
        events["object_0_mass"] = numpy.log(events.LeadPhoton_mass)
        events["object_0_is_pho"] = awkward.ones_like(events["object_0_is_pho"])

        events["object_1_pt"] = numpy.log(events.SubleadPhoton_pt)      
        events["object_1_eta"] = events.SubleadPhoton_eta
        events["object_1_phi"] = events.SubleadPhoton_phi
        events["object_1_mass"] = numpy.log(events.SubleadPhoton_mass)
        events["object_1_is_pho"] = awkward.ones_like(events["object_1_is_pho"])

        # 2. Set jets
        for i in range(1, 1 + self.options["objects"]["n_jets"]):
            idx = (i-1) + self.options["objects"]["n_photons"]

            events["object_%d_pt" % idx] = awkward.where(
                events["jet_%d_pt" % i] > 0.,
                numpy.log(events["jet_%d_pt" % i]),
                DUMMY_VALUE * awkward.ones_like(events["object_%d_pt" % idx])
            )
            events["object_%d_mass" % idx] = awkward.where(
                events["jet_%d_pt" % i] > 0.,
                numpy.log(events["jet_%d_mass" % i]),
                DUMMY_VALUE * awkward.ones_like(events["object_%d_mass" % idx])
            )
            events["object_%d_eta" % idx] = awkward.where(
                events["jet_%d_pt" % i] > 0.,
                events["jet_%d_eta" % i],
                DUMMY_VALUE * awkward.ones_like(events["object_%d_eta" % idx])
            )
            events["object_%d_phi" % idx] = awkward.where(
                events["jet_%d_pt" % i] > 0.,
                events["jet_%d_phi" % i],
                DUMMY_VALUE * awkward.ones_like(events["object_%d_phi" % idx])
            )
            events["object_%d_btag" % idx] = awkward.where(
                events["jet_%d_pt" % i] > 0.,
                events["jet_%d_btagDeepFlavB" % i],
                DUMMY_VALUE * awkward.ones_like(events["object_%d_btag" % idx])
            )


            events["object_%d_charge" % idx] = DUMMY_VALUE * awkward.ones_like(events["object_%d_charge" % idx])
            events["object_%d_is_jet" % idx] = awkward.where(
                events["jet_%d_pt" % i] > 0.,
                awkward.ones_like(events["object_%d_is_jet" % idx]), 
                awkward.zeros_like(events["object_%d_is_jet" % idx])
            )

        # 3. Set leptons
        for i in range(1, 1 + self.options["objects"]["n_leptons"]):
            idx = (i-1) + self.options["objects"]["n_photons"] + self.options["objects"]["n_jets"]
            
            events["object_%d_pt" % idx] = awkward.where(
                events["lepton_%d_pt" % i] > 0.,
                numpy.log(events["lepton_%d_pt" % i]),
                DUMMY_VALUE * awkward.ones_like(events["object_%d_pt" % idx])
            )
            events["object_%d_mass" % idx] = awkward.where(
                events["lepton_%d_pt" % i] > 0.,
                numpy.log(events["lepton_%d_mass" % i]),
                DUMMY_VALUE * awkward.ones_like(events["object_%d_mass" % idx])
            )
            events["object_%d_eta" % idx] = awkward.where(
                events["lepton_%d_pt" % i] > 0.,
                events["lepton_%d_eta" % i],
                DUMMY_VALUE * awkward.ones_like(events["object_%d_eta" % idx])
            )
            events["object_%d_phi" % idx] = awkward.where(
                events["lepton_%d_pt" % i] > 0.,
                events["lepton_%d_phi" % i],
                DUMMY_VALUE * awkward.ones_like(events["object_%d_phi" % idx])
            )
            events["object_%d_charge" % idx] = awkward.where(
                events["lepton_%d_pt" % i] > 0.,
                events["lepton_%d_charge" % i],
                DUMMY_VALUE * awkward.ones_like(events["object_%d_charge" % idx])
            )

            events["object_%d_is_ele" % idx] = awkward.where(
                events["lepton_%d_id" % i] == 11,
                awkward.ones_like(events["object_%d_is_ele" % idx]),
                events["object_%d_is_ele" % idx]
            )
            events["object_%d_is_muo" % idx] = awkward.where(
                events["lepton_%d_id" % i] == 13,
                awkward.ones_like(events["object_%d_is_muo" % idx]),
                events["object_%d_is_muo" % idx]
            )
            events["object_%d_is_tau" % idx] = awkward.where(
                events["lepton_%d_id" % i] == 15,
                awkward.ones_like(events["object_%d_is_tau" % idx]),
                events["object_%d_is_tau" % idx]
            )

        # 4. Set MET
        events["object_%d_is_met" % (self.n_objects)] = awkward.ones_like(events["object_%d_is_met" % (self.n_objects)])
        events["object_%d_pt" % (self.n_objects)] = numpy.log(events.MET_pt) 
        events["object_%d_phi" % (self.n_objects)] = events.MET_phi 

        object_fields = [x for x in events.fields if "object" in x]
        objects = events[object_fields]

        # objects shape [n_events, n_objects, n_object_features]
        fields = ["pt", "eta", "phi", "mass", "charge", "btag", "is_met", "is_pho", "is_ele", "is_muo", "is_tau", "is_jet"]
        self.objects = numpy.empty((self.n_events["total"], self.n_objects + 1, len(fields)))
        for i in range(self.n_objects + 1):
            for j, field in enumerate(fields):
                self.objects[:,i,j] = objects["object_%d_%s" % (i, field)]


    def set_targets(self, events):
        """

        """
        gen_particles = ["HggHiggs", "HbbHiggs", "HwwHiggs", "HttHiggs", "Top_1", "Top_2"]
        for x in gen_particles:
            events["target_has_%s" % x] = awkward.where(
                    events["Gen%s_pt" % x] > 0.,
                    awkward.ones_like(events.process_id),
                    awkward.zeros_like(events.process_id)
            )
            for f in ["pt", "eta", "phi", "mass"]:
                array = events["Gen%s_%s" % (x, f)]
                if f in ["pt", "mass"]:
                    array = numpy.log(array)
                events["target_%s_%s" % (x, f)] = awkward.where(
                        events["target_has_%s" % x],
                        array,
                        DUMMY_VALUE * awkward.ones_like(events.process_id)
                )

        events["target_met_pt"] = numpy.log(events.GenMET_pt)
        events["target_met_phi"] = events.GenMET_phi

        target_fields = [x for x in events.fields if "target" in x]
        self.targets = events[target_fields]


    def write_data(self):
        """

        """
        os.system("mkdir -p %s" % self.output_dir)
        self.targets["objects"] = self.objects

        awkward.to_parquet(self.targets, self.output_dir + "/data.parquet")
