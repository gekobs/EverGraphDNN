import awkward
import numpy
import json
import os

import logging
logger = logging.getLogger(__name__)

from evergraph.utils.misc_utils import update_dict

DUMMY_VALUE = -999.

DEFAULT_OPTIONS = {
        "objects" : {
            "n_jets" : 8,
            "n_leptons" : 4,
            "n_photons" : 2,
            "n_met" : 1
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
    :param options: dictionary of options for prepping events
    :type options: dict
    :param short: flag to just process 10k events
    :type short: bool
    :param selection: apply a selection to a specific phase space (ttHH, HH->ggbb, HH->ggTauTau)
    :type selection: str
    """
    def __init__(self, input_dir, output_dir, options = {}, objects = None, short = False, selection = None, **kwargs):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.options = update_dict(original = DEFAULT_OPTIONS, new = options) 

        self.short = short
        self.selection = selection 
    
        if self.selection is not None:
            if self.selection not in ["ttHH", "Hadronic"]:
                logger.exception("[DataPrepper : __init__] Selection '%s' is not currently supported." % (self.selection))
                raise ValueError()

        if objects is None:
            self.objects = "photons,leptons,jets,met"
        else:
            self.objects = objects

        self.do_photons = False
        self.do_leptons = False
        self.do_jets = False
        self.do_met = False

        logger.debug("[DataPrepper : __init__] Processing the following objects:")
        if "photons" in self.objects:
            self.do_photons = True
            logger.debug("\tPhotons")
        if "leptons" in self.objects:
            self.do_leptons = True
            logger.debug("\tLeptons")
        if "jets" in self.objects:
            self.do_jets = True
            logger.debug("\tJets")
        if "met" in self.objects:
            self.do_met = True
            logger.debug("\tMET")

        if not self.do_photons:
            self.options["objects"]["n_photons"] = 0
        if not self.do_leptons:
            self.options["objects"]["n_leptons"] = 0
        if not self.do_jets:
            self.options["objects"]["n_jets"] = 0
        if not self.do_met:
            self.options["objects"]["n_met"] = 0


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

        if self.selection is not None:
            events = self.apply_selection(events)

        if self.short:
            idx = numpy.random.randint(low = 0, high = self.n_events["total"], size = 100000)
            events = events[idx]
            self.n_events["total"] = len(events)

        for proc, proc_id in self.process_map.items():
            events_proc = events[events.process_id == proc_id]
            self.n_events[proc] = len(events_proc)

            logger.debug("\t %s : %d events (%.1f percent)" % (proc, self.n_events[proc], 100. * float(self.n_events[proc]) / float(self.n_events["total"])))

        return events


    def apply_selection(self, events):
        """

        """
        if self.selection == "ttHH":
            hadronic = (events.n_lep_tau == 0) & (events.n_jets >= 4)
            semilep = (events.n_lep_tau >= 1) & (events.n_jets >= 2)
            multilep = (events.n_lep_tau >= 2)

            tthh_ggbb = events.process_id == self.process_map["ttHH_ggbb"]
            gen_cuts = (events.GenHggHiggs_pt > 0) & (events.GenTop_1_pt > 0) & (events.GenTop_2_pt > 0) & (events.GenHbbHiggs_pt > 0)

            cut = (hadronic | semilep | multilep) & tthh_ggbb & gen_cuts
            
            
        elif self.selection == "HHggbb":
            hh_ggbb = events.process_id = self.proces_map["HH_ggbb"]
            hadronic = (events.n_lep_tau == 0) & (events.n_jets >= 2) & (events.n_jets <= 3)
            cut = hadronic & hh_ggbb

        elif self.selection == "Hadronic":
            tthh_ggbb = events.process_id == self.process_map["ttHH_ggbb"]
            tth = events.process_id == self.process_map["ttH_M125"]
            proc_cut = tthh_ggbb | tth

            hadronic = (events.n_lep_tau == 0) & (events.n_jets >= 4)

            cut = proc_cut & hadronic

        events = events[cut]

        logger.debug("[DataPrepper : apply_selection] After applying selection '%s', there are %d events (eff. of %.2f percent)." % (self.selection, len(events), (100. * float(len(events))) / float(self.n_events["total"])))
        self.n_events["total"] = len(events)

        return events
 

    def set_objects(self, events):
        """

        """
        self.n_objects = self.options["objects"]["n_jets"] + self.options["objects"]["n_leptons"] + self.options["objects"]["n_photons"] + self.options["objects"]["n_met"]
        for i in range(self.n_objects):
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
        if self.do_photons:
            events["object_0_pt"] = numpy.log(events.LeadPhoton_pt) 
            events["object_0_eta"] = events.LeadPhoton_eta
            events["object_0_phi"] = events.LeadPhoton_phi
            events["object_0_mass"] = awkward.zeros_like(events.LeadPhoton_pt) 
            events["object_0_is_pho"] = awkward.ones_like(events["object_0_is_pho"])

            events["object_1_pt"] = numpy.log(events.SubleadPhoton_pt)      
            events["object_1_eta"] = events.SubleadPhoton_eta
            events["object_1_phi"] = events.SubleadPhoton_phi
            events["object_1_mass"] = awkward.zeros_like(events.LeadPhoton_pt) 
            events["object_1_is_pho"] = awkward.ones_like(events["object_1_is_pho"])

        # 2. Set jets
        if self.do_jets:
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
        if self.do_leptons:
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
        if self.do_met:
            events["object_%d_is_met" % (self.n_objects - 1)] = awkward.ones_like(events["object_%d_is_met" % (self.n_objects - 1)])
            events["object_%d_pt" % (self.n_objects - 1)] = numpy.log(events.MET_pt) 
            events["object_%d_phi" % (self.n_objects - 1)] = events.MET_phi 

        object_fields = [x for x in events.fields if "object" in x]
        objects = events[object_fields]

        # objects shape [n_events, n_objects, n_object_features]
        #fields = ["pt", "eta", "phi", "mass", "charge", "btag", "is_met", "is_pho", "is_ele", "is_muo", "is_tau", "is_jet"]
        fields = ["pt", "eta", "phi", "btag", "mass"]
        self.objects = numpy.empty((self.n_events["total"], self.n_objects, len(fields)))
        for i in range(self.n_objects):
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
        os.system("mkdir -p %s/data" % self.output_dir)
        self.targets["objects"] = self.objects

        # shuffle events
        self.targets = self.targets[numpy.random.permutation(len(self.targets))]

        n = 100000 # 100k events per file
        n_files = int(len(self.targets) / n)
        if n_files % n != 0:
            n_files += 1

        for i in range(n_files):
            if i == n_files - 1:
                array = self.targets[i*n:]
            else:
                array = self.targets[i*n:(i+1)*n]
            awkward.to_parquet(array, self.output_dir + "/data/file%d.parquet" % i)

        awkward.to_parquet.dataset("%s/data" % self.output_dir)
