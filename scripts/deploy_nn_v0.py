from argparse import ArgumentParser
import descarteslabs as dl
import json
import numpy as np
import os
from tqdm import tqdm


def deploy_tpa_nn(dlkey, product_id, model_name):
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    import descarteslabs as dl
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras import layers
    import matplotlib.pyplot as plt


    s2_id = 'sentinel-2:L1C'
    s2cloud_id = 'sentinel-2:L1C:dlcloud:v1'
    start_datetime = '2015-01-01'
    end_datetime = '2021-01-01'
    band_descriptions = {
        'coastal-aerosol': 'Aerosols, 442nm',
        'blue': 'Blue, 492nm',
        'green': 'Green, 559nm',
        'red': 'Red, 665nm',
        'red-edge': 'Red Edge 1, 704nm',
        'red-edge-2': 'Red Edge 2, 739nm',
        'red-edge-3': 'Red Edge 3, 779nm',
        'nir': 'NIR, 833nm',
        'red-edge-4': 'Red Edge 4, 864nm',
        'water-vapor': 'Water Vapor, 943nm',
        'swir1': 'SWIR 1, 1610nm',
        'swir2': 'SWIR 2, 2186nm'
    }

    month_interval = 4
    month_step = 2


    def get_imagery(dlkey):
        imagery = dict()
        tile = dl.scenes.DLTile.from_key(dlkey)

        s2_scenes, ctx = dl.scenes.search(products=s2_id, start_datetime=start_datetime, aoi=tile, limit=None)
        s2cloud_scenes, _ = dl.scenes.search(products=s2cloud_id, start_datetime=start_datetime, aoi=tile, limit=None)

        s2_keys = [s.properties.key for s in s2_scenes]
        s2cloud_keys = [s.properties.key for s in s2cloud_scenes]

        # match S2/cloud products
        s2_scenes = s2_scenes.filter(lambda i: i.properties.key in s2cloud_keys)
        s2cloud_scenes = s2cloud_scenes.filter(lambda i: i.properties.key in s2_keys)

        # compute median composites per month
        s2_bands = list(band_descriptions.keys())

        sd = datetime(2019, 1, 1)
        ed = datetime(2020, 12, 1)

        this_start = sd
        this_end = this_start + relativedelta(months=month_interval)
        process = True
        while process == True:
            print(this_start)
            print(this_end)
            print('')

            if this_end >= ed:
                process = False

            s2_scenes_filt = s2_scenes.filter(lambda i: ((i.properties.date >= this_start) &
                                                        (i.properties.date <= this_end)))
            s2cloud_scenes_filt = s2cloud_scenes.filter(lambda i: ((i.properties.date >= this_start) &
                                                                   (i.properties.date <= this_end)))

            s2_stack, raster_info = s2_scenes_filt.stack(bands=s2_bands,
                                                         ctx=ctx,
                                                         resampler='cubic',
                                                         scaling='raw',
                                                         flatten=['properties.date.year',
                                                                  'properties.date.month',
                                                                  'properties.date.day'],
                                                         raster_info=True,
                                                         bands_axis=-1)

            s2cloud_stack = s2cloud_scenes_filt.stack(bands=['mask_cloud', 'mask_cloudshadow'],
                                                      ctx=ctx,
                                                      scaling='raw',
                                                      flatten=['properties.date.year',
                                                               'properties.date.month',
                                                               'properties.date.day'],
                                                      bands_axis=-1)

            cloud_mask = s2cloud_stack[:, :, :, 0] > 0
            shadow_mask = s2cloud_stack[:, :, :, 1] > 0
            s2_mask = np.expand_dims(np.logical_or(cloud_mask, shadow_mask), axis=-1)
            s2_mask = np.repeat(s2_mask, s2_stack.shape[-1], axis=-1)
            s2_stack.mask = np.logical_or(s2_stack.mask, s2_mask)

            s2_median = np.ma.median(s2_stack, axis=0)
            this_date = s2_scenes_filt[0].properties.date.strftime('%Y-%m-%d')
            imagery[this_date] = s2_median

            this_start += relativedelta(months=month_step)
            this_end += relativedelta(months=month_step)

        return imagery, raster_info


    def get_pixel_vectors(image):
        pixel_vectors = list()
        width, height, channels = image.shape
        for i in range(width):
            for j in range(height):
                pixel_vector = list()
                for bidx in range(channels):
                    pixel_vector.append(image[i, j, bidx])
                pixel_vectors.append(pixel_vector)

        return pixel_vectors, width, height


    def normalize(x):
        return (np.array(x) - 0) / (3000 - 0)


    # grab all imagery
    imagery, raster_info = get_imagery(dlkey)

    # load model
    dl.Storage().get_file(model_name, model_name)
    model = keras.models.load_model(model_name)

    # make predictions
    preds_stack = list()
    for month in imagery.keys():
        this_image = imagery[month]
        patch = normalize(this_image)
        patch=patch[:223, :223, :]
        #print("patch shape", np.expand_dims(patch, axis=0).shape)
        #print(np.min(patch), np.max(patch))
        #test_pixel_vectors, width, height = get_pixel_vectors(this_image)
        #test_pixel_vectors = normalize(test_pixel_vectors)
        preds = model.predict(np.expand_dims(patch, axis=0))
        #print(preds)
        #plt.imshow(np.stack((patch[:,:,3],
        #                     patch[:,:,2],
        #                     patch[:,:,1]), axis=-1))
        #plt.show()
        preds_img = np.ones(patch.shape[:2]) * preds[0][1]

        preds_stack.append(preds_img)

    # take median across all predictions
    preds_stack = np.stack(preds_stack, axis=0)
    preds_stack = np.ma.array(data=preds_stack, mask=np.isnan(preds_stack))

    preds_median = np.ma.median(preds_stack, axis=0)
    preds_mean = np.ma.mean(preds_stack, axis=0)
    preds = np.ma.stack([preds_median, preds_mean], axis=0)

    # upload results to catalog
    product = dl.catalog.Product.get(product_id)
    image_upload = dl.catalog.Image(product=product, name=dlkey.replace(':', '_'))
    image_upload.acquired = '2020-01-01'

    upload = image_upload.upload_ndarray(preds.filled(-1),
                                         raster_meta=raster_info[0],
                                         overwrite=True,
                                         overviews=[2, 4, 8, 16, 32, 64, 128, 256, 512],
                                         overview_resampler=dl.catalog.OverviewResampler.AVERAGE)
    #upload.wait_for_completion()
    #print(upload.status)


def set_up_model(model_file, model_name):
    # check if the model exists in storage
    if dl.Storage().exists(model_name):
        print('Model {} found in DLStorage'.format(model_name))
    else:
        dl.Storage().set_file(model_name, model_file)
        print('Model {} uploaded to DLStorage with key {}'.format(model_file, model_name))


def get_tiles_from_roi(roi_file, tilesize, pad):
    with open(roi_file, 'r') as f:
        fc = json.load(f)
        try:
            features = fc['features']
        except:
            features = fc['geometries']

    all_keys = list()
    ctr =0
    for feature in features:
        for tile in dl.Raster().iter_dltiles_from_shape(10.0, tilesize, pad, feature):
            all_keys.append(tile['properties']['key'])
            ctr +=1
            print(ctr, end='\r')

    print('Split ROI into {} tiles'.format(len(all_keys)))

    return all_keys


def get_product(create_product, product_id, product_name, product_desc):
    if create_product:
        product = dl.catalog.Product(id=product_id,
                                     name=product_name,
                                     description=product_desc)
        product.save()

        band_name1 = 'median'
        band1 = dl.catalog.SpectralBand(name=band_name1, product=product)
        band1.data_type = dl.catalog.DataType.FLOAT32
        band1.data_range = (0, 1)
        band1.display_range = (0, 1)
        band1.band_index = 0
        band1.nodata = -1
        band1.save()

        band_name2 = 'mean'
        band2 = dl.catalog.SpectralBand(name=band_name2, product=product)
        band2.data_type = dl.catalog.DataType.FLOAT32
        band2.data_range = (0, 1)
        band2.display_range = (0, 1)
        band2.band_index = 1
        band2.nodata = -1
        band2.save()

        print('Created product and bands')

    else:
        product = dl.catalog.Product.get(id='earthrise:' + product_id)
        print('Got product')

    return product


def main(args):
    # get catalog product
    product = get_product(args.create_product,
                          args.product_id,
                          args.product_name,
                          args.product_desc)

    # grab tiles (DLKeys) from specified ROI
    tiles = get_tiles_from_roi(args.roi_file, args.tilesize, args.pad)

    # set up model in DLStorage
    set_up_model(args.model_file, args.model_name)

    image = 'us.gcr.io/dl-ci-cd/images/tasks/public/py3.8:v2020.09.22-5-ga6b4e5fa'
    name = args.product_name
    cpus = 1
    memory = '6Gi'
    maximum_concurrency = 60
    retry_count = 0
    task_timeout = 20000

    # create async function that runs in Tasks
    #deploy_tpa_nn(tiles[100], product.id, args.model_name
    async_func = dl.Tasks().create_function(deploy_tpa_nn,
                                            image=image,
                                            name=name,
                                            cpus=cpus,
                                            memory=memory,
                                            maximum_concurrency=maximum_concurrency,
                                            retry_count=retry_count,
                                            task_timeout=task_timeout)

    # run this function over every tile
    for dlkey in tqdm(tiles):
        task = async_func(dlkey, product_id=product.id, model_name=args.model_name)
        #deploy_tpa_nn(dlkey, product_id=product.id, model_name=args.model_name)
        #print(task.log)

if __name__ == "__main__":
    parser = ArgumentParser('Configure TPA detector deployment')

    parser.add_argument('--roi_file', type=str, help='GeoJSON file with ROI to deploy over', default='../data/bali.json')
    parser.add_argument('--tilesize', type=int, help='Tilesize in pixels', default=512)
    parser.add_argument('--pad', type=int, help='Padding in pixels', default=16)
    parser.add_argument('--create_product', help='Create new catalog product', action='store_true')
    parser.add_argument('--product_id', type=str, help='ID of catalog product', default='tpa_nn_toa')
    parser.add_argument('--product_name', type=str, help='Name of catalog product', default='TPA NN TOA')
    parser.add_argument('--product_desc', type=str, help='Description of catalog product', default='test')
    parser.add_argument('--model_file', type=str, help='Model file for prediction',
                        default='../models/model_filtered_toa-12-09-2020.h5')
    parser.add_argument('--model_name', type=str, help='Model name in DLStorage',
                        default='model_filtered_toa-12-09-2020.h5')

    args = parser.parse_args()

    main(args)
