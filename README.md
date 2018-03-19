# A Voronoi Based Method for Extracting Features from Crystals for Quantum Machine Learning

This is my solution to the NOMAD 2018 Transparent Conducting Oxide Kaggle, where we were asked to predict formation energies and bandgaps of candidate sesquioxide materials.  This work was inspired by the work of [Ward, et al.](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.024104), who has produced a similar package in Java using Magpie.  This package will reproduce the features used in my best solution, Private Leaderboard 0.0567 #11, with a simple XGBRegressor.  Simply modify the loading commands in the beginning of the script to load an .ase format crystal structure and produce a pandas dataframe with features.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Currently tested with:

```
python 3.6.4
numpy 1.13.3
scipy 1.0.0
pandas 0.22.0
networkx 2.1
tess 0.1.3
```

## Running the scripts

```
python TesselateandFeaturize.py 
```

Yields Pandas dataframe with features for all the training structures from the NOMAD 2018 Kaggle.

## Authors

* **Karl Hujsak** - *Initial work* - [khujsak](https://github.com/khujsak)


## Acknowledgments

* Logan Ward
