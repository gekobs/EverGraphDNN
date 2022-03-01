import awkward
import vector
import numpy

vector.register_awkward()

import logging
logger = logging.getLogger(__name__)

from higgs_dna.taggers.tagger import Tagger, NOMINAL_TAG
from higgs_dna.selections import object_selections, lepton_selections, jet_selections, tau_selections, fatjet_selections, gen_selections
from higgs_dna.utils import awkward_utils, misc_utils

DUMMY_VALUE = -999.
DEFAULT_OPTIONS = {
    "electrons" : {
        "pt" : 7.0,
        "eta" : 2.5,
        "dxy" : 0.045,
        "dz" : 0.2,
        "id" : "WP90",
        "dr_photons" : 0.2,
        "veto_transition" : True
    },
    "muons" : {
        "pt" : 5.0,
        "eta" : 2.4,
        "dxy" : 0.045,
        "dz" : 0.2,
        "id" : "medium",
        "pfRelIso03_all" : 0.3,
        "dr_photons" : 0.2
    },
    "taus" : {
        "pt" : 18.0,
        "eta" : 2.3,
        "dz" : 0.2,
        "deeptau_vs_ele" : 1,
        "deeptau_vs_mu" : 2,
        "deeptau_vs_jet" : 4,
        "dr_photons" : 0.2,
        "dr_electrons" : 0.2,
        "dr_muons" : 0.2
    },
    "fatjets" : {
        "pt" : 250.,
        "eta" : 2.4,
        "dr_photons" : 0.4,
        "dr_electrons" : 0.4,
        "dr_muons" : 0.4,
        "dr_taus" : 0.4,
    }, 
    "jets" : {
        "pt" : 25.0,
        "eta" : 2.4,
        "looseID" : True,
        "dr_photons" : 0.4,
        "dr_electrons" : 0.4,
        "dr_muons" : 0.4,
        "dr_taus" : 0.4,
    },
    "z_veto" : [80., 100.]
}

class EverGraphTagger(Tagger):
    """
    ttHH->ggXX tagger
    """

    def __init__(self, name, options = {}, is_data = None, year = None):
        super(EverGraphTagger, self).__init__(name, options, is_data, year)

        if not options:
            self.options = DEFAULT_OPTIONS
        else:
            self.options = misc_utils.update_dict(
                    original = DEFAULT_OPTIONS,
                    new = options
            )
 

    def calculate_selection(self, events):
        # Electrons
        electron_cut = lepton_selections.select_electrons(
                electrons = events.Electron,
                options = self.options["electrons"],
                clean = {
                    "photons" : {
                        "objects" : events.Diphoton.Photon,
                        "min_dr" : self.options["electrons"]["dr_photons"]
                    }
                },
                name = "SelectedElectron",
                tagger = self
        )

        electrons = awkward_utils.add_field(
                events = events,
                name = "SelectedElectron",
                data = events.Electron[electron_cut]
        )

        # Muons
        muon_cut = lepton_selections.select_muons(
                muons = events.Muon,
                options = self.options["muons"],
                clean = {
                    "photons" : {
                        "objects" : events.Diphoton.Photon,
                        "min_dr" : self.options["muons"]["dr_photons"]
                    }
                },
                name = "SelectedMuon",
                tagger = self
        )

        muons = awkward_utils.add_field(
                events = events,
                name = "SelectedMuon",
                data = events.Muon[muon_cut]
        )

        # Taus
        tau_cut = tau_selections.select_taus(
                taus = events.Tau,
                options = self.options["taus"],
                clean = {
                    "photons" : {
                        "objects" : events.Diphoton.Photon,
                        "min_dr" : self.options["taus"]["dr_photons"]
                    },
                    "electrons" : {
                        "objects" : events.SelectedElectron,
                        "min_dr" : self.options["taus"]["dr_electrons"]
                    },
                    "muons" : {
                        "objects" : events.SelectedMuon,
                        "min_dr" : self.options["taus"]["dr_muons"]
                    }
                },
                name = "AnalysisTau",
                tagger = self
        )

        taus = awkward_utils.add_field(
                events = events,
                name = "AnalysisTau",
                data = events.Tau[tau_cut]
        )

        # Fat jets
        fatjet_cut = fatjet_selections.select_fatjets(
                fatjets = events.FatJet,
                options = self.options["fatjets"],
                clean = {
                    "photons" : {
                        "objects" : events.Diphoton.Photon,
                        "min_dr" : self.options["jets"]["dr_photons"]
                    },
                    "electrons" : {
                        "objects" : events.SelectedElectron,
                        "min_dr" : self.options["jets"]["dr_electrons"]
                    },
                    "muons" : {
                        "objects" : events.SelectedMuon,
                        "min_dr" : self.options["jets"]["dr_muons"]
                    },
                    "taus" : {
                        "objects" : events.AnalysisTau,
                        "min_dr" : self.options["jets"]["dr_taus"]
                    }
                },
                name = "SelectedFatJet",
                tagger = self
        )

        fatjets = awkward_utils.add_field(
                events = events,
                name = "SelectedFatJet",
                data = events.FatJet[fatjet_cut]
        )       

        # Jets
        jet_cut = jet_selections.select_jets(
                jets = events.Jet,
                options = self.options["jets"],
                clean = {
                    "photons" : {
                        "objects" : events.Diphoton.Photon,
                        "min_dr" : self.options["jets"]["dr_photons"]
                    },
                    "electrons" : {
                        "objects" : events.SelectedElectron,
                        "min_dr" : self.options["jets"]["dr_electrons"]
                    },
                    "muons" : {
                        "objects" : events.SelectedMuon,
                        "min_dr" : self.options["jets"]["dr_muons"]
                    },
                    "taus" : {
                        "objects" : events.AnalysisTau,
                        "min_dr" : self.options["jets"]["dr_taus"]
                    }
                },
                name = "SelectedJet",
                tagger = self
        )

        jets = awkward_utils.add_field(
                events = events,
                name = "SelectedJet",
                data = events.Jet[jet_cut]
        )

        bjets = jets[awkward.argsort(jets.btagDeepFlavB, axis = 1, ascending = False)]

        ### Z veto ###
        ee_pairs = awkward.combinations(electrons, 2, fields = ["LeadLepton", "SubleadLepton"])
        mumu_pairs = awkward.combinations(muons, 2, fields = ["LeadLepton", "SubleadLepton"])
        dilep_pairs = awkward.concatenate(
                [ee_pairs, mumu_pairs],
                axis = 1
        )

        lead_lep_p4 = vector.awk({
            "pt" : dilep_pairs.LeadLepton.pt,
            "eta" : dilep_pairs.LeadLepton.eta,
            "phi" : dilep_pairs.LeadLepton.phi,
            "mass" : dilep_pairs.LeadLepton.mass
        })
        sublead_lep_p4 = vector.awk({
            "pt" : dilep_pairs.SubleadLepton.pt,
            "eta" : dilep_pairs.SubleadLepton.eta,
            "phi" : dilep_pairs.SubleadLepton.phi,
            "mass" : dilep_pairs.SubleadLepton.mass
        })
        z_candidates = lead_lep_p4 + sublead_lep_p4

        os_cut = dilep_pairs["LeadLepton"].charge * dilep_pairs["SubleadLepton"].charge == -1
        z_mass_cut = (z_candidates.mass > self.options["z_veto"][0]) & (z_candidates.mass < self.options["z_veto"][1])

        z_veto = ~(os_cut & z_mass_cut) # z-veto on individual z candidates (in principle, can be more than 1 per event)
        z_veto = (awkward.num(z_candidates) == 0) | (awkward.any(z_veto, axis = 1)) # if any z candidate in an event fails the veto, the event is vetoed. If the event does not have any z candidates, we do not veto
 

        ### Presel step 3: construct di-tau candidates and assign to category ###
        taus = awkward.with_field(taus, awkward.ones_like(taus.pt) * 15, "id")
        electrons = awkward.with_field(electrons, awkward.ones_like(electrons.pt) * 11, "id")
        muons = awkward.with_field(muons, awkward.ones_like(muons.pt) * 13, "id")

        leptons = awkward.concatenate(
            [taus, electrons, muons],
            axis = 1
        )
        leptons = leptons[awkward.argsort(leptons.pt, ascending = False, axis = 1)]


        for objects, name in zip([leptons, jets, fatjets], ["lepton", "jet", "fatjet"]):
            awkward_utils.add_object_fields(
                    events = events,
                    name = name,
                    objects = objects,
                    n_objects = 8 if name == "jet" else 4,
                    dummy_value = DUMMY_VALUE
            )


        awkward_utils.add_field(events, ("Diphoton", "pt_mgg"), events.Diphoton.pt / events.Diphoton.mass)
        awkward_utils.add_field(events, ("LeadPhoton", "pt_mgg"), events.LeadPhoton.pt / events.Diphoton.mass)
        awkward_utils.add_field(events, ("SubleadPhoton", "pt_mgg"), events.SubleadPhoton.pt / events.Diphoton.mass)

        # Gen info
        if not self.is_data:
            gen_hbb = gen_selections.select_x_to_yz(events.GenPart, 25, 5, 5)
            gen_hww = gen_selections.select_x_to_yz(events.GenPart, 25, 24, 24)
            gen_htt = gen_selections.select_x_to_yz(events.GenPart, 25, 15, 15)

            gen_tops = gen_selections.select_x_to_yz(events.GenPart, 6, 5, 24)

            awkward_utils.add_object_fields(events, "GenHbbHiggs", gen_hbb.GenParent, n_objects = 1, fields = ["pt", "eta", "phi", "mass"])
            awkward_utils.add_object_fields(events, "GenHwwHiggs", gen_hww.GenParent, n_objects = 1, fields = ["pt", "eta", "phi", "mass"])
            awkward_utils.add_object_fields(events, "GenHttHiggs", gen_htt.GenParent, n_objects = 1, fields = ["pt", "eta", "phi", "mass"])
            awkward_utils.add_object_fields(events, "GenTop", gen_top.GenParent, n_objects = 2)

        # Preselection
        n_electrons = awkward.num(electrons)
        awkward_utils.add_field(events, "n_electrons", n_electrons)

        n_muons = awkward.num(muons)
        awkward_utils.add_field(events, "n_muons", n_muons)

        n_leptons = n_electrons + n_muons
        awkward_utils.add_field(events, "n_leptons", n_leptons)

        n_taus = awkward.num(taus)
        awkward_utils.add_field(events, "n_taus", n_taus)

        n_lep_tau = n_leptons + n_taus
        awkward_utils.add_field(events, "n_lep_tau", n_lep_tau)

        n_jets = awkward.num(jets)
        awkward_utils.add_field(events, "n_jets", n_jets)

        n_fatjets = awkward.num(fatjets)
        awkward_utils.add_field(events, "n_fatjets", n_fatjets)

        pho_idmva_cut = (events.LeadPhoton.mvaID > -0.7) & (events.SubleadPhoton.mvaID > -0.7)

        presel_cut = ((n_lep_tau >= 1) | (n_jets >= 2) | (n_fatjets >= 1)) & z_veto & pho_idmva_cut 

        self.register_cuts(
            names = ["presel cut"], 
            results = [presel_cut]
        )

        return presel_cut, events
    



