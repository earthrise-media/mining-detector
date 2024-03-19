# Gold Mine Detector

Code for the automated detection of artisanal gold mines in Sentinel-2 satellite imagery, with links to related journalism. The data are presented at [Amazon Mining Watch](https://amazonminingwatch.org).

<!--![mining-header](https://user-images.githubusercontent.com/13071901/146877405-3ec46c73-cc80-4b1a-8ad1-aeb189bb0b38.jpg)-->
[![mining-header-planet](https://user-images.githubusercontent.com/13071901/146877590-b083eace-2084-4945-b739-0f8dda79eaa9.jpg)](https://amazonminingwatch.org)

Quick links: 
* [**!! MARCH 2024 DATA AND MODEL UPDATES**](https://github.com/earthrise-media/mining-detector#2023-data-and-model-updates)
* [**INTERPRETING THE FINDINGS**](https://github.com/earthrise-media/mining-detector#interpreting-the-findings)
* [**JOURNALISM**](https://github.com/earthrise-media/mining-detector#journalism)
* [**METHODOLOGY**](https://github.com/earthrise-media/mining-detector#methodology)
* [**MINING**](https://github.com/earthrise-media/mining-detector#results) AND [**AIRSTRIPS**](https://github.com/earthrise-media/mining-detector#clandestine-airstrips-and-airstrips-dataset) DATASETS

---
## Data and model updates (March 2024)

Development of the mining detector halted in 2022 when we lost access to the geospatial computing platform at Descartes Labs. With the arrival of [new API methods](https://medium.com/google-earth/pixels-to-the-people-2d3c14a46da6) to export pixels from Google Earth Engine (GEE), we were able to swap GEE in for Descartes Labs as image source. The original Amazon Mining Watch survey was built on 2020 Sentinel-2 composite satellite imagery. With the redevelopment comes:

* [Yearly assessments of mining activity for 2018-2023](https://github.com/earthrise-media/mining-detector#results). 
* A new Sentinel-2 satellite data pipeline based on Google Earth Engine. Anyone with a GEE account should be able to [run this code](https://github.com/earthrise-media/mining-detector#running-the-code).
* New [models](https://github.com/earthrise-media/mining-detector#models). While preserving the original model architecture, we trained from scratch using the GEE data, with added positive and negative data sampling based on model evaluations and our improved understanding of the scope of mining activities in the Amazon basin. 

Mining expanded each year in the study period. New mining devestated parts of Yanomami, Kayapó, and Munduruku indigenous territories, despite legal prohibitions, and it continues to spread from giant mine fields and into scattered and more remote regions of the Amazon rainforest. Recently, operators of large mining dredges have begun pumping sediments directly from river banks and bottoms, as can be seen in the data in the ravaged riverbanks of Rio Puré and Rio Boia in western Amazonas state, Brazil. 


## Interpreting the findings

The mining of concern here touches every country in the Amazon basin. In the typical process, miners slash the rainforest to bare earth and then pump water through underlying sediments to liberate the minerals. They introduce mercury to form an amalgam with the gold, to separte it from other particles, and later they burn off the mercury to arrive at a fairly pure gold metal. This type of mining is called _artisanal_ because it is practiced by small groups of individuals with some machinery, such as pumps, dredges, and excavators. The mining proceeds along streams and rivers, which provide water and access into the rainforest.

The environmental and human costs are high. Mining transforms healthy rainforest into a wasteland of bare earth and toxic sediment pools. Mercury enters adjacent streams and rivers. In the Amazon basin, miners frequently operate within indigenous lands, bringing in unfamiliar diseases and sometimes violent conflict. 

Scars from the mining can be seen from satellite. On the banks of a river, you will observe jumbled, multi-colored wastewater pools. They can be brown, tan, yellow, different shades of green, even turquoise. For the most part they are irregular in size, shape, and orientation. Often nearby you can observe miners' encampments, perhaps with blue-tarped tents, and in well-developed mines, a dirt airstrip cut to fly in miners and to fly out the gold. 

On Amazon Mining Watch, detected mines are delineated by the yellow stroke. Here are some characteristic examples of mines:

![MinesEx](https://user-images.githubusercontent.com/11287904/150804841-fabcef8f-4394-46ff-be11-c87ad789ae19.jpg)
(These are mines.)

The automated detector is a work in progress. With limited bootstrap sampling, we extrapolated signficantly to run over the whole of the Amazon basin. There are some false detections, and we encourage users to apply discretion in interpreting the findings. Terrain features that can masquerade as mines include sandbars in rivers, braided rivers, farm ponds, and aquaculture ponds (two examples below), like so:

<!--![NotMinesEx2](https://user-images.githubusercontent.com/11287904/150863564-0b861bef-5cb0-4ea7-bc8e-440b20bece03.jpg)-->
![NotMinesEx](https://user-images.githubusercontent.com/11287904/150816991-7ca7c55f-1c27-460f-bfec-bbdd3e2146ed.jpg)
(These are _not_ mines.)

You can recognize aquaculture ponds by their geometric shape, efficient use of space, and presence in agricultural zones. 

From the March 2024 data release, we note in particular some false positives from aquaculture and other wet industrial operations around Manaus and an area of landslides in hilly terrain of southern Loreto, Peru.

A more common model error is the _false negative_, where the model fails to detect a mine or the full extent of a mine. 

Mine scars where the rainforest has begun to heal may not be detected in later years, so mined area both expands and recedes over time. We can see some virtue in this unintended feature of the models, and we decided not to consider it a false negative nor to mitigate against it. 

On the whole, false detections are relatively few given how widespread the mining is, and we hope this will be a useful resource to those interested in tracking mining activity in the region. 

#### Basemap Imagery

Mining in the Amazon is expanding rapidly, and frequent cloud cover makes it challenging to stitch together comprehensive satellite basemaps. In the Amazon mine map, you will sometimes see healthy rainforest in areas where mining activity is indicated. In that case, the displayed imagery is out of date. (To make for a better user experience, the imagery displayed is different from the imagery used for detection.) 

We provide two display options for the web map. The [Mapbox satellite basemap](https://www.mapbox.com/) is the default. It provides detailed, sub-meter resolution views of many of the mines. The second option is the newly published [Sentinel-2 basemap](https://www.maptiler.com/news/2022/01/free-access-to-10m-global-satellite-map/) from MapTiler, which uses imagery from 2020 and 2021 exclusively, but at 10-meter resolution. In the example below, mine detections are displayed over the Mapbox basemap at left and over the MapTiler Sentinel-2 basemap at right. 

![MapboxvsSentinel2basemaps](https://user-images.githubusercontent.com/11287904/150791417-c431cd40-3d02-4c13-be70-06adc8a29ac1.jpg)

For up-to-date views, we recommend searching the full Sentinel-2 catalog on [SentinelHub EO Browser](https://apps.sentinel-hub.com/eo-browser/?zoom=14&lat=-7.13214&lng=-57.36245&visualizationUrl=https%3A%2F%2Fservices.sentinel-hub.com%2Fogc%2Fwms%2Fbd86bcc0-f318-402b-a145-015f85b9427e&datasetId=S2L2A&fromTime=2020-09-16T00%3A00%3A00.000Z&toTime=2020-09-16T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR) or the Planetscope data made available through the [Planet Labs NICFI program](https://www.planet.com/nicfi/).

#### Detection Accuracy

Creating quantitative accuracy metrics for a system like this is not always easy or constructive. For example, if the system asserted that there are no mines at all in the Amazon basin, it would be better than 99% accurate, because such a large proportion of the landscape remains unmined.

To provide a more constructive measure, we validated a random subsample of the system's detections. This allows us to estimate what is known as the precision or positive predictive value for the classifier. In essence, it tells you the likelihood that box marked as a mine is actually a mine. On our latest run, we see a precision of 98.2%. For a sample of 500 mining detections, you can expect to see about 9 misclassifications. In our sample, a third of the false detections still identified mining activity, but mining for materials such as bauxite rather than gold.

#### Area estimation

The goal of this work is mine detection rather than area estimation, and our classification operates on 480 m x 480 m patches. If the network determines that mining exists within the patch, then the full patch is declared a mine. This leads to a systematic overestimation of mined area if it is naively computed from the polygon boundaries. Building a segmentation model to delineate mine boundaries could be a useful extension of this work.

## Journalism 

![MiningTitlesCollage](https://user-images.githubusercontent.com/11287904/150589512-5d2f1e1c-b946-4f35-90a0-09efbcecc83a.jpg)

This work grew out of a series of collaborations with journalists and with advocates at Survival International seeking to expose illegal gold mining activity and document its impacts on the environment and on local indigenous communities. We began identifying mines by sight in satellite imagery. Later, some high school classes helped sift through images. Finally it made sense to try to automate the identification of mine sites. The training datasets for the machine-learned models followed from those initial human surveys.

#### Reports using the automated detections
* [Las pistas illegales que bullen en la selva Venezolana](https://elpais.com/internacional/2022-01-30/las-pistas-clandestinas-que-bullen-en-la-selva-venezolana.html), from _El País_ and [ArmandoInfo](https://armando.info/la-mineria-ilegal-monto-sus-bases-aereas-en-la-selva/), 2022. First in the series [Corredor Furtivo](https://armando.info/series/corredor-furtivo/). Produced in conjunction with the Pulitzer Center's Rainforest Investigation Network ([in English, translated](https://pulitzercenter.org/stories/illegal-mining-set-air-bases-jungle-spanish)).
* [The pollution of illegal gold mining in the Tapajós River](https://infoamazonia.org/en/storymap/the-pollution-of-illegal-gold-mining-in-the-tapajos-river/), _InfoAmazonia_, 2021. The story is part of a series, [Murky Waters](https://infoamazonia.org/en/project/murky-waters/), on pollution in the Amazon River system.
* [Novas imagens de satélite revelam garimpo ainda mais destruidor na TI Yanomami](https://reporterbrasil.org.br/2023/02/novas-imagens-de-satelite-revelam-garimpo-ainda-mais-destruidor-na-ti-yanomami/), on new expansion of illegal mining in Yanomami Indigenous Territory, _Rapórter Brasil_, 2023.
* [Suspected leader of the so called narcogarimpos extracted gold from environmental area without the permission of Brazilian regulation authority](https://reporterbrasil.org.br/2023/10/suspected-leader-of-the-so-called-narcogarimpos-extracted-gold-from-environmental-area-without-the-permission-of-brazilian-regulation-authority/), part of the [Narcogarimpos](https://narcogarimpos.reporterbrasil.org.br/en/) investigation from _Repórter Brasil_, 2023.

#### Clandestine airstrips and airstrips dataset

Rough dirt airstrips, often cut illegally from the forest and unregistered with authorities, allow miners to access the mines and to fly out the gold. The Intercept Brasil and The New York Times surveyed over a thousand clandestine airstrips in Brazil's Legal Amazon, identifying 362 landing strips within 20 kilometers of mining activity. The inquiry into the airstrips' role in the expansion of mining led to a pair of stories and a short documentary film: 

* [The illegal airstrips bringing toxic mining to Brazil’s indigenous land](https://www.nytimes.com/interactive/2022/08/02/world/americas/brazil-airstrips-illegal-mining.html), _The New York Times_, 2022.
* [As pistas da destruição](https://theintercept.com/2022/08/02/amazonia-pistas-clandestinas-garimpo/), _The Intercept_, 2022. 
* [Os pilotos da Amazônia](https://www.youtube.com/watch?v=IA-Rk_hdl4M), _The Intercept_, short film, 2022.

The airstrip location data are [available for download](data/airstrips/). The clandestine airstrips dataset is the result of a collaborative reporting effort by The Intercept Brasil, The New York Times, and the Rainforest Investigations Network, an initiative of The Pulitzer Center. The Intercept Brasil created the project within the network, which was later joined by The New York Times. The data were gathered by Earthrise Media from OpenStreetMap and from satellite images of Amazônia Legal in 2021, augmented with input from the Socio-Environmental Institute of Brazil, the Yanomami Hutukara Association, and government reports, and verified by the newsrooms.

#### Related reporting on open-pit mining
* [Empresa de Nova York tem ligação com contrabando de ouro ilegal da Amazônia](https://reporterbrasil.org.br/2023/04/empresa-de-nova-york-tem-ligacao-com-contrabando-de-ouro-ilegal-da-amazonia/), from _Repórter Brasil_ and [NBC News](https://www.nbcnews.com/news/two-new-yorkers-tried-leave-brazil-77-pounds-gold-luggage-rcna67221), 2023. Report on links between a New York company, gold smuggling, and rainforest destruction in Kayapó indigenous land. 
* [Garimpo destruidor](https://theintercept.com/2021/12/04/garimpo-ilegal-sai-cinza-para-amazonia/), _The Intercept_, 2021. Video of a helicopter flyover of mine devastation.
* [Gana por ouro](https://theintercept.com/2021/09/16/mineradora-novata-ja-explorou-32-vezes-mais-ouro-do-que-o-previsto-em-area-protegida-da-amazonia/),  _The Intercept_, 2021. Report on an industrial gold mine operating without proper environmental permits. Two weeks after the story appeared the mine was shut down and fined. The mine [continued to operate](https://www.intercept.com.br/2022/03/26/presidente-ibama-pressionou-subalterno-para-liberar-mineradora-de-ouro-embargada/) in defiance of the embargo, 2022.
* [Serious risk of attack by miners on uncontacted Yanomami in Brazil](https://www.survivalinternational.org/news/12655), Survival International, 2021.
* [Illegal mining sparks malaria outbreak in indigenous territories in Brazil](https://infoamazonia.org/en/2020/11/25/mineracao-ilegal-contribui-para-surto-de-malaria-em-terras-indigenas-no-para/), _InfoAmazonia_ and _Mongabay_, 2020.
* [Amazon gold rush: The threatened tribe](https://graphics.reuters.com/BRAZIL-INDIGENOUS/MINING/rlgvdllonvo/index.html), _Reuters_, 2019, on illegal mining in protected Yanomami Indigenous Territory.

Many thanks to the journalists whose skill and resourceful reporting brought these important stories to light.

## Methodology

### Overview

The mine detector is a lightweight convolutional neural network, which we train to discriminate mines from other terrain by feeding it hand-labeled examples of mines and other key features as they appear in Sentinel-2 satellite imagery. The network operates on square patches of data extracted from the [Sentinel 2 L1C data product](https://sentinel.esa.int/web/sentinel/missions/sentinel-2). Each pixel in the patch captures the light reflected from Earth's surface in twelve bands of visible and infrared light. We average (median composite) the Sentinel data across a period of many months to reduce the presence of clouds, cloud shadow, and other transitory effects. 

During run time, the network assesses each patch for signs of recent mining activity, and then the region of interest is shifted by half a patch width for the network to make a subsequent assessment. This process proceeds across the entire region of interest. The network makes over 100 million individual assessments in covering the 6.7 million square kilometers of the Amazon basin. 

The system was developed for use in the Amazon, but it has also been seen to work in other tropical biomes.

### Results

#### Yearly asessment of mining in the Amazon basin, 2018-2023 (v2 Amazon Mining Watch dataset)

This most recent assessment was run with an [ensemble of six models](https://github.com/earthrise-media/mining-detector/blob/main/models/48px_v3.2-3.7ensemble_2024-02-13.h5), and all patches with mean score over 0.5 were recorded. 

[Output data](https://github.com/earthrise-media/mining-detector/tree/main/data/outputs/48px_v3.2-3.7ensemble) are saved year by year and presented in two formats. The first records each saved patch along with the mean score and the six individual predictions from models 3.2-3.7. The second format, with filenames tagged _dissolved-0.6_, saves only patches meeting a higher 0.6 mean score threshold and then merges adjacent patches into larger polygons. 

The dissolved predictions are presented on [Amazon Mining Watch](https://amazonminingwatch.org/) and should suffice for most users. At lower prediction threshold, the ensemble is more expansive, capturing more mining at the cost of more false positive detections; at higher threshold, the ensemble is stingy and more likely to be correct in the mines it surfaces. The choice of 0.6 reflects our own preference within this tradeoff. Users wanting to tune the prediction threshold can work with the data in the patch format. 

#### Assessement of mining in the Amazon basin in 2020 (v1 Amazon Mining Watch dataset)

[Amazon mine map](https://earthrise-media.github.io/mining-detector/amazon-mine-map.html) and the [output dataset](data/outputs/44px_v2.9/mining_amazon_all_unified_thresh_0.8_v44px_v2.6-2.9_2020-01-01_2021-02-01_period_4_method_median.geojson). This data was largely generated with the [44px v2.6 model](models/44px_v2.6_2021-11-09.h5). A small portion in the Brazillian state of Pará was analyzed using the [44px v2.9 model](models/44px_v2.9_2022-02-28.h5) to improve accuracy.

#### Tapajós basin mining progression, 2016-2020 

[Tapajós mine map](https://earthrise-media.github.io/mining-detector/tapajos-mining-2016-2020pub.html) and [output dataset](data/outputs/28_px_v9/28_px_tapajos_2016-2020_thresh_0.5.geojson). In this case, we analyzed the region yearly from 2016-2020 to monitor the growth of mining in the area, using the earlier [28px v9 model](models/28_px_v9.h5). 

#### Hand-validated dectections of mines in Venezuela's Bolívar and Amazonas states in 2020

[Venezuela mine map](https://earthrise-media.github.io/mining-detector/bolivar-amazonas-2020v9verified.html), [Bolívar dataset](data/outputs/28_px_v9/bolivar_2020_thresh_0.8verified.geojson) and [Amazonas dataset](data/outputs/28_px_v9/amazonas_2020_thresh_0.5verified.geojson). Analysis via the 28px v9 model. 

#### Generalization Test in Ghana's Ashanti Region, 2018 and 2020

[Ghana mine map](https://earthrise-media.github.io/mining-detector/ghana-ashanti-2018-2020-v2.8.html) and [output dataset](data/outputs/44px_v2.8/mining_ghana_ashanti_v44px_v2.8_2017-2020.geojson). This was a test of the model's ability to generalize to tropical geographies outside of the Amazon basin, using the [44px v2.8 model](https://github.com/earthrise-media/mining-detector/blob/main/models/44px_v2.8_2021-11-11.h5). 
 
### Running the Code
This repo contains all code needed to generate data, train models, and deploy a model to predict presence of mining in a region of interest. While we welcome external development and use of the code, subject to terms of an open [MIT license](https://github.com/earthrise-media/mining-detector/blob/eboyda-patch-1/LICENSE), creating datasets and deploying the model currently requires access to the [Descartes Labs](https://descarteslabs.com/) platform. 

#### Setup

Model inference ran on Python 3.9 in the anaconda3-2022.05 environment with Tensorflow XX.X. 

#### Notebooks
The system runs from three core notebooks. 

##### `create_dataset.ipynb` (requires Descartes Labs access)
Given a GeoJSON file of sampling locations, generate a dataset of Sentinel 2 images. Dataset is stored as a pickled list of numpy arrays.

##### `train_model.ipynb`
Train a neural network based on the images stored in the `data/training_data/` directory. Data used to train this model is stored at `s3://mining-data.earthrise.media`.

##### `deploy_model.ipynb` (requires Descartes Labs access)
Given a model file and a GeoJSON describing a region of interest, run the model and download the results. Options exist to deploy the model on a directory of ROI files.

#### Data
- `data/boundaries` contains GeoJSON polygon boundaries for regions of interest where the model has been deployed.
- `data/sampling_locations` contains GeoJSON datasets that are used as sampling locations to generate training datasets. Datasets in this directory should be considered "confirmed," and positive/negative class should be indicated in the file's title.

#### Models
The models directory contains keras neural network models saved as `.h5` files. The model names indicate the patch size evaluated by the model, followed by the model's version number and date of creation. Each model file is paired with a corresponding config `.txt` file that logs the datasets used to train the model, some hyperparameters, and the model's performance on the test dataset.

The model `44px_v2.8_2021-11-11.h5` is currently the top performer overall, though some specificity has been sacrificed for generalization. Different models have different strengths/weaknesses. There are also versions of model v2.6 that operate on [RGB](44px_v2.6_rgb_2021-11-11.h5) and [RGB+IR](models/44px_v2.6_rgb_ir_2021-11-11.h5) data. These may be of interest when evaluating whether multispectral data from Sentinel is required.

### License

The code in this repository are available for reuse under an open [MIT License](https://github.com/earthrise-media/mining-detector/blob/main/LICENSE). The data is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). In publication, please cite Earth Genome, with reference to this repository.
