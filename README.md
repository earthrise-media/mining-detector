# mining-detector

## Setup

Download and install [Conda env](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) if not already installed:
```
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
```

Next, create the conda environment named `mining-detector` by running `conda env create` from the repo root directory. Code has been developed and tested on a Mac with python version 3.9.7. Other platforms and python releases may work, but have not yet been tested.

If desired, the data used for model training may be accessed on s3 at `s3://mining-data.earthrise.media`.

## Notebooks
