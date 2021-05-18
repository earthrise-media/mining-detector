import descarteslabs as dl
import geojson
import json
from argparse import ArgumentParser

parser = ArgumentParser('Configure Data Download Parameters')
parser.add_argument('--roi_file', type=str, help='GeoJSON file with ROI to deploy over')
parser.add_argument('--product_id', type=str, help='ID of catalog product')

args = parser.parse_args()

product_id = args.product_id
fn = args.roi_file

with open('../data/' + fn, 'r') as f:
  geo_fc = json.load(f)

region = geo_fc['features'][0]['geometry']

fc_id = [fc.id for fc in dl.vectors.FeatureCollection.list() if product_id in fc.id][0]
fc = dl.vectors.FeatureCollection(fc_id)

fc_filtered = fc.filter(geometry = region)

features = []

for feat in fc_filtered.features():
  features.append(feat.geojson)

feature_collection = geojson.FeatureCollection(features)

filepath = '../outputs/' + product_id + '.geojson'

with open(filepath, 'w') as f:
  geojson.dump(feature_collection, f)
