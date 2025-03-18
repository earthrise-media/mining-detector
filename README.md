# Gold Mine Detector

Code for the automated detection of artisanal gold mines in Sentinel-2 satellite imagery, with links to related journalism. The data are presented at [Amazon Mining Watch](https://amazonminingwatch.org).

<!--![mining-header](https://user-images.githubusercontent.com/13071901/146877405-3ec46c73-cc80-4b1a-8ad1-aeb189bb0b38.jpg)-->
[![mining-header-planet](https://user-images.githubusercontent.com/13071901/146877590-b083eace-2084-4945-b739-0f8dda79eaa9.jpg)](https://amazonminingwatch.org)

Quick links: 
* [**!! MARCH 2024 DATA AND MODEL UPDATES**](https://github.com/earthrise-media/mining-detector#march-2024-data-and-model-updates)
* [**INTERPRETING THE FINDINGS**](https://github.com/earthrise-media/mining-detector#interpreting-the-findings)
* [**JOURNALISM**](https://github.com/earthrise-media/mining-detector#journalism)
* [**METHODOLOGY**](https://github.com/earthrise-media/mining-detector#methodology)
* [**MINING**](https://github.com/earthrise-media/mining-detector#results) AND [**AIRSTRIPS**](https://github.com/earthrise-media/mining-detector#clandestine-airstrips-and-airstrips-dataset) DATASETS

---
## March 2024 data and model updates

Development of the mining detector halted in 2022 when we lost access to the geospatial computing platform at Descartes Labs. With the arrival of [new API methods](https://medium.com/google-earth/pixels-to-the-people-2d3c14a46da6) to export pixels from Google Earth Engine (GEE), we were able to swap GEE in for Descartes Labs as image source. The original Amazon Mining Watch survey was built on 2020 composite Sentinel-2 satellite imagery. With the redevelopment comes:

* [Yearly assessments of mining activity for 2018-2023](https://github.com/earthrise-media/mining-detector#results). 
* A new Sentinel-2 satellite data pipeline based on Google Earth Engine. Anyone with a GEE account should be able to [run this code](https://github.com/earthrise-media/mining-detector/blob/main/gee/README.md).
* New [models](https://github.com/earthrise-media/mining-detector#models). While preserving the original model architecture, we trained from scratch using the GEE data, with added positive and negative data sampling based on model evaluations and our improved understanding of the scope of mining activities in the Amazon basin. 

Mining expanded each year in the study period, notably into previously untouched areas of Yanomami, Kayapó, and Munduruku indigenous territories. It continues to spread into scattered and remote regions of the Amazon rainforest. Even some of the tiniest isolated detections are working mines. In western Amazonas, Brazil, floating dredges are scooping soils from river banks and bottoms in the search for gold, seen in the ravaged riverbanks of Rio Puré and Rio Boia in the most recent years' data.


## Interpreting the findings

The mining of concern here touches every country in the Amazon basin. In the typical process, miners slash the rainforest to bare earth and then pump water through underlying sediments to liberate the minerals. They introduce mercury to form an amalgam with the gold, to separte it from other particles, and later they burn off the mercury to arrive at a fairly pure gold metal. This type of mining is called _artisanal_ because it is practiced by small groups of individuals with some machinery, such as pumps, dredges, and excavators. The mining proceeds along streams and rivers, which provide water and access into the rainforest.

Scars from the mining can be seen from satellite. On the banks of a river, you will observe muddy flats jumbled together with multi-colored toxic wastewater pools. The pools can be brown, tan, yellow, different shades of green, even turquoise. For the most part they are irregular in size, shape, and orientation. Often nearby you can observe miners' encampments, perhaps with blue-tarped tents, and in well-developed mines, a dirt airstrip cut to fly in miners and to fly out the gold. 

On Amazon Mining Watch, detected mines are delineated by the yellow stroke. Here are some characteristic examples of mines:

![MinesEx](https://user-images.githubusercontent.com/11287904/150804841-fabcef8f-4394-46ff-be11-c87ad789ae19.jpg)
(These are mines.)

With limited bootstrap sampling, we extrapolated to run over the whole of the Amazon basin. There are some false detections, and we encourage users to apply discretion in interpreting the findings. Terrain features that can masquerade as mines include sandbars in rivers, braided rivers, farm ponds, and aquaculture ponds, like so:

<!--![NotMinesEx2](https://user-images.githubusercontent.com/11287904/150863564-0b861bef-5cb0-4ea7-bc8e-440b20bece03.jpg)-->
![NotMinesEx](https://user-images.githubusercontent.com/11287904/150816991-7ca7c55f-1c27-460f-bfec-bbdd3e2146ed.jpg)
(These are _not_ mines.)

You can recognize aquaculture ponds by their geometric shape, efficient use of space, and presence in agricultural zones. 

From the March 2024 data release, we note in particular some false positives from aquaculture and other wet industrial operations around Manaus and an area of landslides in hilly terrain of southern Loreto, Peru.

A more common model error is the _false negative_, where the model fails to detect a mine or the full extent of a mine. 

Where the rainforest has begun to heal, mine scars may not be detected in later years, and so mined area both expands and recedes over time. We see some value in this model response and we decided not to correct it. 

On the whole, false detections are relatively few given how widespread the mining is, and we hope this will be a useful resource to those interested in tracking mining activity in the region. 

#### Detection Accuracy

Creating quantitative accuracy metrics for a system like this is not always easy or constructive. For example, if the system asserted that there are no mines at all in the Amazon basin, it would be better than 99% accurate, because such a large proportion of the landscape is not mined. 

To provide one indicative measure, we validated a random sample of 500 detections from 2023. This allows us to estimate what is known as the precision or positive predictive value for the classifier. In essence, it tells you the likelihood that a patch marked as a mine is actually a mine. Of the 500 samples, 498 have artisanal mining scars. One is an industrial mine, and one is a remnant of the construction of the Balbina dam and power station from around 1985. The estimated precision of the classifier in this real-world context is 99.6%. 

#### Area estimation

The goal of this work is mine detection rather than area estimation, and our classification operates on square image patches covering around twenty hectares each. If the network determines that mining exists within the patch, then the full patch is declared a mine. This leads to a systematic overestimation of mined area if it is naively computed from the polygon boundaries. Building a segmentation model to delineate mine boundaries could be a useful extension of this work.

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

The airstrip location data are [available for download](data/airstrips/). The clandestine airstrips dataset is the result of a collaborative reporting effort by The Intercept Brasil, The New York Times, and the Rainforest Investigations Network, an initiative of The Pulitzer Center. The Intercept Brasil created the project within the network, which was later joined by The New York Times. The data were gathered by Earth Genome from OpenStreetMap and from satellite images of Amazônia Legal in 2021, augmented with input from the Socio-Environmental Institute of Brazil, the Yanomami Hutukara Association, and government reports, and verified by the newsrooms.

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

#### Yearly asessment of mining in the Amazon basin, 2018-2024 (v2 Amazon Mining Watch dataset)

This most recent assessment was run with an ensemble of six models: [48px_v3.2-3.7ensemble_2024-02-13.h5](https://github.com/earthrise-media/mining-detector/blob/main/models/48px_v3.2-3.7ensemble_2024-02-13.h5). We recorded outputs for all patches with a mean score over 0.5, on a scale from 0 to 1. 

[Output data](https://github.com/earthrise-media/mining-detector/tree/main/data/outputs/48px_v3.2-3.7ensemble) are saved year by year and presented in three formats. The first format records the mean score and the six individual predictions from models 3.2-3.7 for each saved patch. The second, streamlined, format, with filenames tagged _dissolved-0.6_, saves only patches meeting a higher 0.6 mean score threshold and then merges adjacent patches into larger polygons. 

The dissolved predictions are presented on [Amazon Mining Watch](https://amazonminingwatch.org/) and should suffice for most users. At lower prediction threshold, the ensemble captures more mining at the cost of more false positive detections; at higher threshold, the ensemble is stingier with its predictions and more likely to be correct in the mines it surfaces. The choice of 0.6 reflects our own preference in this tradeoff. Users wanting to tune the prediction threshold can work with the data in the patch format.

Finally, because of year-to-year variance in detections of small mine scars, and because mine scars can fade from detection where vegetation regrows, we include a set of _cumulative_ detections. These datasets aggregate the dissolved yearly detections from 2018 through the later year indicated in the filename, delineating places where mining has ever been detected to that point. By 2023, the cumulative area mapped is almost 50% larger than the area mapped in 2023 alone.

#### Assessement of mining in the Amazon basin in 2020 (v1 Amazon Mining Watch dataset)

[Amazon mine map](https://earthrise-media.github.io/mining-detector/amazon-mine-map.html) and the [output dataset](data/outputs/44px_v2.9/mining_amazon_all_unified_thresh_0.8_v44px_v2.6-2.9_2020-01-01_2021-02-01_period_4_method_median.geojson). This data was largely generated with the [44px v2.6 model](models/44px_v2.6_2021-11-09.h5). A small portion in the Brazillian state of Pará was analyzed using the [44px v2.9 model](models/44px_v2.9_2022-02-28.h5) to improve accuracy.

#### Tapajós basin mining progression, 2016-2020 

[Tapajós mine map](https://earthrise-media.github.io/mining-detector/tapajos-mining-2016-2020pub.html) and [output dataset](data/outputs/28_px_v9/28_px_tapajos_2016-2020_thresh_0.5.geojson). In this case, we analyzed the region yearly from 2016-2020 to monitor the growth of mining in the area, using the earlier [28px v9 model](models/28_px_v9.h5). 

#### Hand-validated dectections of mines in Venezuela's Bolívar and Amazonas states in 2020

[Venezuela mine map](https://earthrise-media.github.io/mining-detector/bolivar-amazonas-2020v9verified.html), [Bolívar dataset](data/outputs/28_px_v9/bolivar_2020_thresh_0.8verified.geojson) and [Amazonas dataset](data/outputs/28_px_v9/amazonas_2020_thresh_0.5verified.geojson). Analysis via the 28px v9 model. 

#### Generalization Tests in Ghana

These runs test the ability of the models to generalize to tropical geographies outside the Amazon basin. The detections could be more comprehensive, but they appear to capture the broad patterns of mining in the country.

[Ghana 2024 dataset](data/outputs/48px_v3.2-3.7ensemble/ghana_48px_v3.2-3.7ensemble_0.50_2024-01-01_2024-11-15-dissolved-0.6.geojson) (January 1 - November 15).

Ashanti region, combined 2017 and 2020 [map](https://earthrise-media.github.io/mining-detector/ghana-ashanti-2018-2020-v2.8.html) and [dataset](data/outputs/44px_v2.8/mining_ghana_ashanti_v44px_v2.8_2017-2020.geojson).
 
### Organization of the repository

This repo contains all code needed to generate data, train models, and deploy a model to predict presence of mining in a region of interest. We welcome external use of the code subject to terms of an open [MIT license](https://github.com/earthrise-media/mining-detector/blob/eboyda-patch-1/LICENSE). 

#### Code

Code for data generation and model inference is in the [`gee`](https://github.com/earthrise-media/mining-detector/tree/main/gee) folder. The [readme](https://github.com/earthrise-media/mining-detector/blob/main/gee/README.md) there provides instructions. 

After training data generation, training runs from [`notebooks/train_model.ipynb`](https://github.com/earthrise-media/mining-detector/blob/main/notebooks/train_model.ipynb). 

#### Data inputs
- `data/boundaries` contains GeoJSON polygon boundaries for regions of interest where the model has been deployed.
- `data/sampling_locations` contains GeoJSON datasets that are used as sampling locations to generate training datasets. A positive/negative class label is indicated in each file's name.

#### Models
The `models` directory contains keras neural network models saved as `.h5` files. The model names indicate the patch size evaluated by the model, followed by the model's version number and date of creation. Each model file is paired with a corresponding config `.txt` file that logs the datasets used to train the model, some hyperparameters, and the model's performance on a test dataset. 

### License

The code in this repository are available for reuse under an open [MIT License](https://github.com/earthrise-media/mining-detector/blob/main/LICENSE). The data is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). In publication, please cite Earth Genome, with reference to this repository.
