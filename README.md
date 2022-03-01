# EverGraphDNN
Generalizable collider physics generator-level event reconstruction with a graph DNN.

## Installation
**1. Clone this repository**
```  
git clone git@github.com:sam-may/EverGraphDNN.git 
cd EverGraphDNN
```
**2. Install dependencies**

The necessary dependencies (listed in ```environment.yml```) can be installed manually, but the suggested way is to create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/mana
ge-environments.html) by running:
``` 
conda env create -f environment.yml
```

**3. Install ```evergraph```**
Suggested way to install is:
```
pip install -e .
```

## Quick start notes
An output file from HiggsDNA after running the selection in `higgs_dna/evergraph_tagger.py` is available in this directory: `/home/users/smay/HiggsDNA/scripts/evergraph_25Jan2022/`.

To prep inputs for DNN training with a hadronic selection (>=4 jets, 0 leptons + taus) and selecting ttH and ttHH->ggbb, run the command:
```
python scripts/prep.py --input_dir "/home/users/smay/HiggsDNA/scripts/evergraph_25Jan2022/" --output_dir "hadronic_1Mar2022" --log-level "DEBUG" --selection "Hadronic" --objects "photons,jets,met"
```
and to train a graph CNN which performs convolutional operations on each pair of input objects from the 2 photons, 8 leading jets, and MET (11 choose 2 = 55 pairs per event), run the following command:
```
python train.py --input_dir "hadronic_1Mar2022/" --output_dir "hadronic_1Mar2022_tthh_vs_tth/" --log-level "DEBUG"
```
this graph CNN will be trained with the following targets:
- whether the event has an H->bb pair or not (`"target_has_HbbHiggs"`)
- the pt and eta of the H->bb Higgs
- the pt and eta of the H->gg Higgs (as a sanity check we can compare this with the pt and eta we get from adding the four vectors of the photons)


TODO items:
- [ ] Make training targets and network details configurable through `json`
- [ ] Develop `evergraph/evaluation` tools for assessing performance of graph CNNs
