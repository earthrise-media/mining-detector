# Amazon Gold Mining Map

Models and code for the automated detection of gold mines in Sentinel-2 satellite imagery; webmapped output of gold mines detected in the Amazon rainforest; links to journalism using this work. 

<!--![mining-header](https://user-images.githubusercontent.com/13071901/146877405-3ec46c73-cc80-4b1a-8ad1-aeb189bb0b38.jpg)-->
[![mining-header-planet](https://user-images.githubusercontent.com/13071901/146877590-b083eace-2084-4945-b739-0f8dda79eaa9.jpg)](https://earthrise-media.github.io/mining-detector/amazon-mine-map.html)

* [**LAUNCH WEB MAP**](https://earthrise-media.github.io/mining-detector/amazon-mine-map.html)
* [**HOW TO INTERPRET THE MAP**](https://github.com/earthrise-media/mining-detector#interpreting-the-map)
* [**METHODOLOGY**](https://github.com/earthrise-media/mining-detector#methodology)
* [**JOURNALISM**]
  
## Interpreting the map



### Users should be aware of the following limitations:

**Old Basemap Imagery**

Mining in the Amazon is growing rapidly. Most basemap imagery in the Amazon is not current, thus some regions classified as containing mines will not appear to have mining activity in the imagery. See example below. Regions in question can be assessed by viewing recent [Sentinel 2 imagery on SentinelHub EO Browser](https://apps.sentinel-hub.com/eo-browser/?zoom=14&lat=-7.13214&lng=-57.36245&visualizationUrl=https%3A%2F%2Fservices.sentinel-hub.com%2Fogc%2Fwms%2Fbd86bcc0-f318-402b-a145-015f85b9427e&datasetId=S2L2A&fromTime=2020-09-16T00%3A00%3A00.000Z&toTime=2020-09-16T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR), or Planetscope data accessible through the [Planet NICFI program](https://www.planet.com/nicfi/).
![mining-imagery-comparison](https://user-images.githubusercontent.com/13071901/146989519-d1e537c4-7d70-438d-b4a5-06b2a41a8482.jpg)

**Model Accuracy**

In order to run across the full breadth of the Amazon basin, the model's sensitivity and precision have been reduced in order to improve generalization. This means that the model outputs have some false positives (mining classification where none is present) and false negatives (mines that are rejected by the model). Common false negative failure modes include older mining sites that may be inactive, and edges of active mining regions. False positives can at times be triggered by natural sedimented pools, man made water bodies, and natural earth-scarring activities such as landslides.

The aggregate assessment of mining status should be trusted, but users should attempt to validate results by eye if precise claims of mined regions are needed. The vast majority of classifications are correct, but we cannot validate each of the detections by hand. Given that mining often happens in clusters, isolated detections of mining should be validated more rigorously for false positives.

Additionally, there are a few regions that we could not assess mining activity by eye in high resolution satellite imagery. We have decided to leave these regions in the output data and maps.
![error types](https://user-images.githubusercontent.com/13071901/147019219-98c518fb-72d1-4e35-bf32-9fe058b5d6eb.jpg)


**Area Overestimation**

The goal of this work is mine detection rather than area estimation, and our classification operates on the classification of 440 m² patches. If the network assesses that mining exists anywhere within the patch, then the full patch is declared a mine. This leads to a systematic overestimation of mined area if it is naively computed from the polygon boundaries. Relative year-to-year change calculations are accurate since the polygon area overestimation is consistent.

Building a segmentation model that operates on detected regions is a viable extension of this work.

## Methodology

The mine detector is a light-weight convolutional neural network, which we train to discrimate mines from other terrain in the Amazon basin by feeding it hand-labeled examples of mines and other key features as they appear in Sentinel-2 satellite imagery. The network operates on 440 m x 440 m patches of data extracted from the [Sentinel 2 L1C data product](https://sentinel.esa.int/web/sentinel/missions/sentinel-2). Each pixel in the patch captures the light reflected from Earth's surface in twelve bands of visible and infrared light. We average (median composite) the Sentinel data across a four-month period to reduce the presence of clouds, cloud shadow, and other transitory effects. 

During run time, the network assesses each patch for signs of recent mining activity, and then the region of interest is shifted by 140 m for the network to make a subsequent assessment. This process proceeds across the entire region of interest. The network makes 326 million individual assessments in covering the 6.7 million square kilometers of the Amazon basin. 

The system was developed for use in the Amazon, but it has also been seen to work in other tropical biomes.

See the discussion of the [code](https://github.com/earthrise-media/mining-detector#running-the-code), below, for more details. 

## Results
### Assessement of mining in the Amazon basin
Mining analysis for the full [Amazon basin](data/boundaries/amazon_basin.geojson) in 2020. The [44px v2.6 model](models/44px_v2.6_2021-11-09.h5) was used for this analysis.

* [2020 map of gold mines in the Amazon basin](https://earthrise-media.github.io/mining-detector/amazon-mine-map.html) 
* [Amazon basin mining dataset (GeoJSON)](data/outputs/44px_v2.6/mining_amazon_all_unified_thresh_0.8_v44px_v2.6_2020-01-01_2021-02-01_period_4_method_median.geojson).

### Tapajós basin mining progression, 2016-2020 
We analyzed the [Tapajos basin](data/boundaries/tapajos_basin.geojson) region yearly from 2016-2020 to monitor the progression of mining in the area. This analysis was run with the earlier [28px v9 model](models/28_px_v9.h5).

Data were published in ["The pollution of illegal gold mining in the Tapajós River"](https://infoamazonia.org/en/storymap/the-pollution-of-illegal-gold-mining-in-the-tapajos-river/), part of InfoAmazonia's series [Murky Waters](https://infoamazonia.org/en/project/murky-waters/), on pollution from waste, mining, and agriculutre in the Amazon River system and links to recent massive sargassum seaweed blooms in the Caribbean.

* [Tapajós mining map, over Mapbox basemap imagery](https://earthrise-media.github.io/mining-detector/tapajos-mining-2016-2020.html)
* [Tapajós mining map, over recent, lower resolution Landsat imagery](https://earthrise-media.github.io/mining-detector/tapajos-mining-2016-2020pub.html)
* [Tapajos mining progression dataset (GeoJSON)](data/outputs/28_px_v9/28_px_tapajos_2016-2020_thresh_0.5.geojson)

### Hand-validated dectections of mines in Venezuela's Bolívar and Amazonas states
We analyzed and hand validated the outputs for the Venezuelan states of Bolivar and Amazonas. This map should not contain false positive sites, though may contain false negatives. This analysis was run with the [28px v9 model](models/28_px_v9.h5).

* [Map of detections](https://earthrise-media.github.io/mining-detector/bolivar-amazonas-2020v9verified.html)
* [Dataset of detections - Bolivar (GeoJSON)](data/outputs/28_px_v9/bolivar_2020_thresh_0.8verified.geojson)
* [Dataset of detections - Amazonas (GeoJSON)](data/outputs/28_px_v9/amazonas_2020_thresh_0.5verified.geojson)

### Generalization Test in Ghana
To test the model's ability to generalize to tropical geographies outside of the Amazon, we ran the [44px v2.8 model](models/44px_v2.8_2021-11-11.h5) across the [Ashanti region of Ghana](data/boundaries/ghana_ashanti.geojson) in 2018 and 2020. 

* [Map of detections in Ghana](https://earthrise-media.github.io/mining-detector/ghana-ashanti-2018-2020-v2.8.html)
* [Ghanaian detections data (GeoJSON)](data/outputs/44px_v2.8/mining_ghana_ashanti_v44px_v2.8_2017-2020.geojson)

## Running the Code
This repo contains all code needed to generate data, train models, and deploy a model to predict presence of mining in a region of interest. While we welcome external development and use of the code, subject to terms of our open [MIT license](WIP link), creating datasets and deploying the model currently requires access to the [Descartes Labs](https://descarteslabs.com/) platform. 

As of January 21, 2022, the work is on hold, as we seek partners to support development toward an ongoing mine-monitoring alert system and data platform. If you are interested in partnering, please contact us at [info@earthrise.media](mailto:info@earthrise.media).

### Setup

Download and install [Miniforge Conda env](https://github.com/conda-forge/miniforge/) if not already installed:


| OS      | Architecture          | Download  |
| --------|-----------------------|-----------|
| Linux   | x86_64 (amd64)        | [Miniforge3-Linux-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh) |
| Linux   | aarch64 (arm64)       | [Miniforge3-Linux-aarch64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh) |
| Linux   | ppc64le (POWER8/9)    | [Miniforge3-Linux-ppc64le](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-ppc64le.sh) |
| OS X    | x86_64                | [Miniforge3-MacOSX-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh) |
| OS X    | arm64 (Apple Silicon) | [Miniforge3-MacOSX-arm64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) |

Then run 
```
chmod +x ~/Downloads/Miniforge3-{platform}-{architecture}.sh
sh ~/Downloads/Miniforge3-{platform}-{architecture}.sh
source ~/miniforge3/bin/activate
```

Next, create a conda environment named `mining-detector` by running `conda env create -f environment.yml` from the repo root directory. Activate the environment by running `conda activate mining-detector`. Code has been developed and tested on a Mac with python version 3.9.7. Other platforms and python releases may work, but have not yet been tested.

The data used for model training may be accessed and downloaded from `s3://mining-data.earthrise.media`.

### Notebooks
The system runs from three core notebooks. 

#### `create_dataset.ipynb` (requires Descartes Labs access)
Given a GeoJSON file of sampling locations, generate a dataset of Sentinel 2 images. Dataset is stored as a pickled list of numpy arrays.

#### `train_model.ipynb`
Train a neural network based on the images stored in the `data/training_data/` directory. Data used to train this model is stored at `s3://mining-data.earthrise.media`.

#### `deploy_model.ipynb` (requires Descartes Labs access)
Given a model file and a GeoJSON describing a region of interest, run the model and download the results. Options exist to deploy the model on a directory of ROI files.

### Data
The data directory contains two directories.
- `data/boundaries` contains GeoJSON polygon boundaries for regions of interest where the model has been deployed.
- `data/sampling_locations` contains GeoJSON datasets that are used as sampling locations to generate training datasets. Datasets in this directory should be considered "confirmed," and positive/negative class should be indicated in the file's title.
- `data/outputs` will be populated with results from `deploy_model.ipynb` in a folder corresponding to the model version used in the run. By default, it is populated with GeoJSON files from our runs and analyses.

### Models
The models directory contains keras neural network models saved as `.h5` files. The model names indicate the patch size evaluated by the model, followed by the model's version number and date of creation. Each model file is paired with a corresponding config `.txt` file that logs the datasets used to train the model, some hyperparameters, and the model's performance on the test dataset.

The model `44px_v2.8_2021-11-11.h5` is currently the top performer overall, though some specificity has been sacrificed for generalization. Different models have different strengths/weaknesses. There are also versions of model v2.6 that operate on [RGB](44px_v2.6_rgb_2021-11-11.h5) and [RGB+IR](models/44px_v2.6_rgb_ir_2021-11-11.h5) data. These may be of interest when evaluating whether multispectral data from Sentinel is required.

### Docs
This directory does not store docs. Instead, it hosts .html files that are displayed on the repo's github pages site at `http://earthrise-media.github.io/mining-detector/{file_name}.html.`
