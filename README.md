# Gold Mine Detector

Code for the automated detection of artisanal gold mines in Sentinel-2 satellite imagery, with links to related journalism. The data are presented at [amazonminingwatch.org](https://amazonminingwatch.org). Amazon Mining Watch is a partnership betwen the Pulitzer Center's Rainforest Investigations Network, Amazon Conservation Association, and Earth Genome.

<!--![mining-header](https://user-images.githubusercontent.com/13071901/146877405-3ec46c73-cc80-4b1a-8ad1-aeb189bb0b38.jpg)-->
[![mining-header-planet](https://user-images.githubusercontent.com/13071901/146877590-b083eace-2084-4945-b739-0f8dda79eaa9.jpg)](https://amazonminingwatch.org)

Quick links: 
* [**AUGUST 2026 UPDATES**](https://github.com/earthrise-media/mining-detector#august-2026-updates)
* [**INTERPRETING THE FINDINGS**](https://github.com/earthrise-media/mining-detector#interpreting-the-findings)
* [**JOURNALISM**](https://github.com/earthrise-media/mining-detector#journalism)
* [**METHODOLOGY**](https://github.com/earthrise-media/mining-detector#methodology)
* [**RESULTS / DATA**](https://github.com/earthrise-media/mining-detector#results)
* [**AIRSTRIPS**](https://github.com/earthrise-media/mining-detector#clandestine-airstrips-and-airstrips-dataset)

---


Notes on earlier model releases (November 2025 foundation-model experiments, March 2024 CNN rebuild, and prior vintages) remain in git history: see the [README at commit `2ce1738`](https://github.com/earthrise-media/mining-detector/blob/2ce17387445321b4d9cb8292e015386390c853f1/README.md).

## August 2026 updates

In August 2026 we published a full rebuild of Amazon Mining Watch models and data, to improve model sensitivity and to account for monitoring conditions on shorter time intervals (especially, increased cloud cover).

* **Models rebuilt from the ground up.** The production detector is an ensemble of convolutional networks trained from scratch on Sentinel-2 image patches. Details are in the [Methodology](#methodology).

* **Expanded training data.** The labeled set now comprises **23,463** patches (2,968 mine / ~20,495 not-mine), spanning train, validation, and geographic holdout splits. This marks an eight-fold increase in samples over the 2024 training dataset. Sample chips from the training data are in [`data/training_gallery/`](data/training_gallery/).

* **New output data, 2018–present.** Detections have been recomputed yearly for  2018-2025 and quarterly starting in 2025.  The new data provides a consistent time series from the beginning of monitoring and removes the earlier break where older years and newer years came from different model generations.

* **Improved handling of clouds.**  The new model gracefully suppresses cloud artefacts, important for monitoring on shorter time windows. Especially at the start of each year, low-cloud satellite views are typically unavailable over large parts of the Amazon basin.

* **Detections corroborated across time.** A detection is confirmed to the cumulative record when it appears in two successive annual periods. The most recent quarterly data is published alongside, bearing a _provisional_ label. Details are under [Persistence](#persistence).

Published products and download links are summarized under [Results](#results).

## Interpreting the findings

The mining of concern here touches every country in the Amazon basin. In the typical process, miners slash the rainforest to bare earth and then pump water through underlying sediments to liberate the minerals. They introduce mercury to form an amalgam with the gold, to separte it from other particles, and later they burn off the mercury to arrive at a fairly pure gold metal. This type of mining is called _artisanal_ because it is practiced by small groups of individuals with some machinery, such as pumps, dredges, and excavators. The mining proceeds along streams and rivers, which provide water and access into the rainforest.

Scars from the mining can be seen from satellite. On the banks of a river, you will observe muddy flats jumbled together with multi-colored toxic wastewater pools. The pools can be brown, tan, yellow, different shades of green, even turquoise. For the most part they are irregular in size, shape, and orientation. Often nearby you can observe miners' encampments, perhaps with blue-tarped tents, and in well-developed mines, a dirt airstrip cut to fly in miners and to fly out the gold. 

Here are some characteristic examples of mines:

![MinesEx](https://user-images.githubusercontent.com/11287904/150804841-fabcef8f-4394-46ff-be11-c87ad789ae19.jpg)
(These are mines.)

With limited bootstrap sampling, we extrapolated to run over the whole of the Amazon basin. There are some false detections, and we encourage users to apply discretion in interpreting the findings. Terrain features that can masquerade as mines include sandbars in rivers, braided rivers, farm ponds, and aquaculture ponds, like so:

<!--![NotMinesEx2](https://user-images.githubusercontent.com/11287904/150863564-0b861bef-5cb0-4ea7-bc8e-440b20bece03.jpg)-->
![NotMinesEx](https://user-images.githubusercontent.com/11287904/150816991-7ca7c55f-1c27-460f-bfec-bbdd3e2146ed.jpg)
(These are _not_ mines.)

You can recognize aquaculture ponds by their geometric shape, efficient use of space, and presence in agricultural zones. 

A more common model error is the _false negative_, where the model fails to detect a mine or the full extent of a mine. 

Where the rainforest has begun to heal, mine scars may not be detected in later years, and so mined area both expands and recedes over time. We see some value in this model response and we decided not to correct it. 

On the whole, false detections are relatively few given how widespread the mining is, and we hope this will be a useful resource to those interested in tracking mining activity in the region. 

#### Detection accuracy

The Amazon basin is enormous: for each period the model scores on the order of a hundred million patches. By contrast, the labeled evaluation sets consists of thousand examples. Metrics on those labels are useful, but they are not a full census of performance across Amazonia. See [Model training and evaluation](#model-training-and-evaluation) for how we report **sensitivity** (share of labeled mines found) and **specificity** (share of labeled non-mines correctly rejected), and for the numbers attached to the recommended single-period product.

#### Area estimation

The primary goal of this work is to detect mines, and our classification operates on square image patches covering around twenty hectares each. Area estimates are refined with a fine-tuned [SAM2](https://ai.meta.com/research/sam2/) segmentation model on RGB Sentinel-2 imagery (see [Methodology](#area-estimation-sam2)). Raster masks are published on [Source Cooperative](https://source.coop/earthgenome/amazon-mining-watch). Further fine-tuning on hard cases remains ongoing.

## Journalism 

![MiningTitlesCollage](https://user-images.githubusercontent.com/11287904/150589512-5d2f1e1c-b946-4f35-90a0-09efbcecc83a.jpg)

This work grew out of a series of collaborations with journalists and with advocates at Survival International seeking to expose illegal gold mining activity and document its impacts on the environment and on local indigenous communities. We began identifying mines by sight in satellite imagery. Later, some high school classes helped sift through images. Finally it made sense to try to automate the identification of mine sites. The training datasets for the machine-learned models followed from those initial human surveys.

#### Selected reporting using the automated detections
* [Las pistas illegales que bullen en la selva Venezolana](https://elpais.com/internacional/2022-01-30/las-pistas-clandestinas-que-bullen-en-la-selva-venezolana.html), from _El País_ and [ArmandoInfo](https://armando.info/la-mineria-ilegal-monto-sus-bases-aereas-en-la-selva/), 2022. First in the series [Corredor Furtivo](https://armando.info/series/corredor-furtivo/). Produced in conjunction with the Pulitzer Center's Rainforest Investigation Network ([in English, translated](https://pulitzercenter.org/stories/illegal-mining-set-air-bases-jungle-spanish)).
* [The pollution of illegal gold mining in the Tapajós River](https://infoamazonia.org/en/storymap/the-pollution-of-illegal-gold-mining-in-the-tapajos-river/), _InfoAmazonia_, 2021. The story is part of a series, [Murky Waters](https://infoamazonia.org/en/project/murky-waters/), on pollution in the Amazon River system.
* [Novas imagens de satélite revelam garimpo ainda mais destruidor na TI Yanomami](https://reporterbrasil.org.br/2023/02/novas-imagens-de-satelite-revelam-garimpo-ainda-mais-destruidor-na-ti-yanomami/), on new expansion of illegal mining in Yanomami Indigenous Territory, _Rapórter Brasil_, 2023.
* [Suspected leader of the so called narcogarimpos extracted gold from environmental area without the permission of Brazilian regulation authority](https://reporterbrasil.org.br/2023/10/suspected-leader-of-the-so-called-narcogarimpos-extracted-gold-from-environmental-area-without-the-permission-of-brazilian-regulation-authority/), part of the [Narcogarimpos](https://narcogarimpos.reporterbrasil.org.br/en/) investigation from _Repórter Brasil_, 2023.
* [Amazon Mining Watch: mapas satelitales confirman nuevos focos de deforestación por actividad minera en países amazónicos](https://convoca.pe/agenda-propia/amazon-mining-watch-mapas-satelitales-confirman-nuevos-focos-de-deforestacion-por), _Convoca_, 2024.
* [Avanço de garimpo em terras indígenas alerta para novos meios de lavagem de ouro](https://reporterbrasil.org.br/2024/07/garimpo-terras-indigenas-alerta-novos-meios-lavagem-ouro/), _Repórter Brasil_, 2024. Also published by _Convoca_ in the [series Dorada Opacidad](https://convoca.pe/doradaopacidad/).
* [Gold mining in the Amazon has doubled in area since 2018, AI tool shows](https://news.mongabay.com/2024/07/gold-mining-in-the-amazon-has-doubled-in-area-since-2018-ai-tool-shows/), _Mongabay_, 2024.

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

The mine detector is an ensemble of convolutional neural networks trained to discriminate mines from other terrain using hand-labeled examples in [Sentinel-2 L1C](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) imagery. Each input is a square multi-spectral patch (~480 m on a side at 10 m resolution; 48×48 pixels × 13 bands). On Google Earth Engine, from the [Cloud Score+ Sentinel-2 harmonized collection](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED), we build median composites over multi-month windows to reduce clouds, cloud shadow, and other short-lived effects.

At run time the ensemble scores patches across the region of interest, stepping by half a patch width so that neighboring assessments overlap. Covering the 8.2 million km² Amazon basin requires over a hundred million patch assessments per period.

The system was developed for the Amazon and has also been observed to transfer to other tropical biomes.

### Model

The July 2026 production model is [`models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5`](models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5): an ensemble of six CNNs trained from scratch on 48×48×13 Sentinel-2 patches. Most members use a ~800k-parameter convolutional architecture (`CNN800k` in `core/model_library.py`); for balance, the ensemble also includes a smaller ~100k-parameter CNN in the lineage of the 2024 architecture. Member scores are averaged to produce the final confidence.

### Model training and evaluation

In addition to a validation split drawn from the labeled pool, we withheld coherent geographic regions from training for model selection and evaluation (see `data/boundaries/geo_holdout_*.geojson`). In particular, the Río Caroní area—covering most of Bolívar state, Venezuela (`geo_holdout_caroni.geojson`)—was unused until training and model selection were complete, so that we can measure how the system generalizes to a region never seen in training.

We report performance with two complementary rates:

* **Sensitivity** (also called *recall*) — of the sites we labeled as mines, what fraction does the system flag? (How completely do we catch known mines?)
* **Specificity** — of the sites we labeled as *not* mines, what fraction does the system correctly leave alone? (How successfully do we reject non-mines?)

For **individual years or quarters**, we recommend the relaxed dual-threshold product under [`postprocessed/`](https://source.coop/earthgenome/amazon-mining-watch/postprocessed) on Source Cooperative. The main confidence cutoff (`t_main = 0.43`) sits at the peak of the F0.5 curve, and isolated candidate patches must clear a higher bar (`t_iso = 0.75`). That is our most deliberate tuning of the catch-mines vs. avoid-false-alarms tradeoff for single-period maps.

On that product:

* **Venezuela holdout:** 1,008 labeled patches (343 mines, 665 not-mines). The model correctly rejected **661 / 665** non-mines (specificity **0.994**) and found **307 / 343** mines (sensitivity **0.895**).
* **Combined validation + two geographic holdouts:** 723 mines and 3,156 not-mines. Specificity **0.998**; sensitivity **0.924**. *(Some of this combined pool informed model or threshold selection.)*

**Caveats.** These figures are not a full account of real-world performance. They measure agreement with sites we labeled, and there is no authoritative ground-truth inventory of mining across Amazonia. Given Sentinel-2’s 10 m resolution, we did not attempt to label or model the smallest operations. Conversely, we tended to label hard cases, and t; `geo_holdout_napo-caquetá.geojson` was later used when training two of the six ensemble members.he model’s strength at rejecting vast areas of intact forest is only partly reflected here.

Architectures we tested before settling on the production ensemble included ~100k-parameter CNNs (the 2024 design), ~800k-parameter CNNs, ResNet-18, and probes on SSL4EO ViT embeddings (class-token only, with 48×48→224×224 rescaling; and CLS + patch-token features at two spatial scales). Ensembling proved essential for reducing noise across base architectures.

### Postprocessing

After inference we apply a dual confidence threshold based on a simple spatial prior: patches with mine scars tend to cluster. Candidate patches must meet a main confidence cutoff; isolated patches (far from other candidates) must meet a higher cutoff. That two-tier rule removes scattershot noise and preserves more true mines than imposing a uniformly strict cutoff.

### Persistence

Amazon Mining Watch emphasizes cumulative detections: the union of mining evidence from the start of monitoring in 2018 through each later date. In our view that best establishes a historical record of lands impacted by mining since monitoring began. But errors accumulate year on year alongside true coverage, so a cumulative needs a defence against false detections that a single-period map does not.

We now require corroboration across time in lieu of the higher confidence level we have applied in the past. A detection enters the record as **confirmed** once it appears in two annual periods within a two-year window; the period recorded is the first of those. A confirmed detection is never withdrawn by a later update.

The most recent periods have no later year to corroborate against. Recent quarterly detections there are marked **provisional** and are held to a stricter confidence threshold in place of that corroboration, then replaced by annual data once the corroborating year arrives.

For 2024, the last period with confirmed detections at time of
writing, the cumulative detections return the following scores with
respect to the evaluation locations used above:

* Venezuela holdout: specificity **0.976**, sensitivity **0.991**
* Combined validation + two geographic holdouts: specificity **0.991**, sensitivity **0.989**

**More caveats.** These numbers are derived from the production
outputs at the locations labeled for evaluation, not a direct run on
image chips from the evaluation datasets.  There are many ways that
distinction can cause labels or predictions to drift: The labels were
drawn against imagery from a specific time period, while the product
accumulates from different landscape views before and after; the
inference grid does not align with the training patches; and by the
nature of a cumulative product, the specificity decreases and the
sensitivity increases as the record grows through time. Still, it is
the best estimator of performance we have for the cumulative
detections set.

For the website we also fold in lower-threshold detections (`t_main = 0.2`) for a few small partner regions in Peru ([`data/boundaries/andes_supplemental.geojson`](data/boundaries/andes_supplemental.geojson)), where the goal is to capture very small-scale mining. We found that we could not train against those sites without driving up false detections elsewhere, indicating a practical lower size limit for a Sentinel-2–based system. 

### Area estimation (SAM2)

To better estimate mined areas, we delineate scars with a fine-tuned [SAM2](https://ai.meta.com/research/sam2/) segmentation model—a clear improvement over estimating area from whole patches or from NDVI masking alone. The same two-year corroboration rule is applied at the pixel level, so a mask pixel enters the cumulative record when it is segmented in two annual periods within a two-year window.

Segmentation is the least mature part of the pipeline. Measured against 184 hand-annotated validation chips, segmented area for a single year mask runs about 1.5× the annotated extent. The bias is largely systematic, which makes comparisons between periods more trustworthy than any single total. The model still needs more fine-tuning on hard cases.

We gratefully acknowledge **Michael Braun**, **Daemon Li**, and **Divas Subedi**, master’s students in the Department of Computer Science at Georgia Tech, for developing the fine-tuned SAM2 segmentation model as part of their course work.

## Results

Historical result tables and maps from earlier model generations are retained in git; see the [Results section of the README at `2ce1738`](https://github.com/earthrise-media/mining-detector/blob/2ce17387445321b4d9cb8292e015386390c853f1/README.md#results).

### August 2026 data products

Bulk outputs are multiple gigabytes and therefore not stored in this repository.  Public copies live on [Source Cooperative — earthgenome/amazon-mining-watch](https://source.coop/earthgenome/amazon-mining-watch). An in-repo catalog of product layout, thresholds, and mirror paths can be found at [`data/outputs/MANIFEST.yaml`](data/outputs/MANIFEST.yaml).

Browsing and download are available on the [source.coop product page](https://source.coop/earthgenome/amazon-mining-watch). Programmatic download of individual files is available via `https://data.source.coop/earthgenome/amazon-mining-watch/<path/to/file>`.

| Product | What it is | On Source Cooperative |
| --- | --- | --- |
| **Detections** | The primary product: one entry per location, with model confidence, the period mining was first confirmed, and confirmed or provisional status. | [`amazon_basin_detections.geojson`](https://data.source.coop/earthgenome/amazon-mining-watch/amazon_basin_detections.geojson) |
| **Scar masks** | Pixelwise mine-scar extent at 10 m Sentinel-2 resolution, from a fine-tuned SAM2 model prompted by the detections above.  Pixel values indicate the time period of onset, e.g. 2024 or 20251 (for Q1 2025). | [`amazon_basin_mining_scar_masks.tif`](https://data.source.coop/earthgenome/amazon-mining-watch/amazon_basin_mining_scar_masks.tif) |
| **Single-period patch detections** | Dual-threshold postprocess for individual years and quarters (`t_main=0.43`, `t_iso=0.75`), unaggregated. Also the set the segmentation model is prompted from. | [`postprocessed`](https://source.coop/earthgenome/amazon-mining-watch/postprocessed) |
| **Raw detections** | Patches with confidence ≥ 0.4 before dual-threshold filtering. Amazon basin and Andes supplemental sit together in one folder. | [`raw_detections`](https://source.coop/earthgenome/amazon-mining-watch/raw_detections) |
| **Archive** | Published January 2026 from a superseded experimental model, retained for the record. Extensively cleaned by hand, both a limitation and a virtue. | [`archived`](https://source.coop/earthgenome/amazon-mining-watch/archived) |

The per-period cumulative files and the stringent `t_main=0.55` single-period set were published earlier in 2026 and have been removed. The former are superseded by the new detections, and the latter now serves only as the provisional-edge threshold described under [Persistence](#persistence).

Model file (in this repo): [`models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5`](models/48px_v4.10b-18d-20g-21a-22bc-ensemble.h5).

### Organization of the repository

This repo contains code to generate data, train models, and run inference over a region of interest. 

#### Code

Data generation and model inference live under [`core/`](core/) (see [`core/README.md`](core/README.md)).

#### Data inputs
- `data/boundaries` — region-of-interest polygons (Amazon ACA, Andes supplemental, geographic holdouts, etc.).
- `data/sampling_locations` — labeled sampling locations used to build training sets.

#### Models
The `models` directory holds Keras `.h5` checkpoints. Names encode patch size, version, and often training date. Each model pairs with a `*_config*.txt` log of datasets, hyperparameters, and held-out metrics, where available.

### License

The code in this repository is available for reuse under an open [MIT License](LICENSE). The data is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). In publication, please cite Earth Genome, with reference to this repository.
