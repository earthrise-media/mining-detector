import argparse

from dateutil.relativedelta import relativedelta
import descarteslabs as dl
from tqdm import tqdm

from scripts import dl_utils


DL_SYSTEM_PARAMS = {
    'image': ('us.gcr.io/dl-ci-cd/images/tasks/public/' +
              'py3.8:v2020.09.22-5-ga6b4e5fa'),
    'cpus': 1,
    'maximum_concurrency': 60,
    'memory': '24Gi',
    'retry_count': 4,
    'task_timeout': 20000,
    'include_modules': ['scripts.dl_utils']
}

def run_model(dlkey, **kwargs):
    """Wrap a call to DescartesRun for use in DL async processing.

    Required kwargs:
        product_id: String ID of a DL catalog product
        model_name: Model name in DL storage
        start_date, end_date: Isoformat date strings

    Optional kwargs are passed to the instantiation of DescartesRun.
    """
    import dl_utils
    runner = dl_utils.DescartesRun(**kwargs)
    runner(dlkey, kwargs['start_date'], kwargs['end_date'])

def main(*args):
    """Deploy a model on the Descartes Labs platform.

    Args:
        args:list: Can include any pair of form (flag, argument) passed to
            the argument parser, e.g. ['--roi_file', '../data/bali.json'].
            Cannot be None if calling from an interpreter. Give [] instead.
    """
    parser = argparse.ArgumentParser('Configure TPA detector deployment')
    parser.add_argument('--roi_file',
                        type=str,
                        help='GeoJSON file with ROI to deploy over',
                        default='../data/bali.json')
    parser.add_argument('--patch_product_id',
                        type=str,
                        help='ID of catalog product',
                        default='')
    parser.add_argument('--product_name',
                        type=str,
                        help='Name of catalog product',
                        default='TPA NN TOA')
    parser.add_argument('--tilesize',
                        type=int,
                        help='Tilesize in pixels',
                        default=840)
    parser.add_argument('--pad',
                        type=int,
                        help='Padding in pixels',
                        default=16)
    parser.add_argument('--patch_model_file',
                        type=str,
                        help='Local path to model file to upload',
                        default='')
    parser.add_argument('--patch_model_name',
                        type=str,
                        help='Model name in DL Storage',
                        default='')
    parser.add_argument('--patch_stride',
                        type=int,
                        help='Stride width in pixels for patch classifier',
                        default=None)
    parser.add_argument('--mosaic_period',
                        type=int,
                        help='Months over which to mosaic image data',
                        default=1)
    parser.add_argument('--mosaic_method',
                        type=str,
                        help='Compositing method: "median"/"min"/"min_masked"',
                        default='min')
    parser.add_argument('--spectrogram_interval',
                        type=int,
                        help=('Spectrogram time interval, in mosaic periods'),
                        default=6)
    # Note on dates: Date range should be longer than the spectrogram length.
    # Starting on successive mosaic periods (typically: monthly), as many
    # spectrograms are created as fit in the date range.
    parser.add_argument('--start_date',
                        type=str,
                        help='Isoformat start date for predictions',
                        default='2020-06-01')
    parser.add_argument('--end_date',
                        type=str,
                        help='Isoformat end date for predictions',
                        default='2020-10-01'),
    parser.add_argument('--run_local',
                        action='store_true',
                        help='Run model locally rather than async on DL.')
    args = parser.parse_args(*args)

    tiles = dl_utils.get_tiles_from_roi(args.roi_file, args.tilesize, args.pad)

    # This init handles product creation and model upload.
    runner = dl_utils.DescartesRun(**vars(args))

    if args.run_local:
        for dlkey in tqdm(tiles):
            runner(dlkey, args.start_date, args.end_date)
    else:
        async_func = dl.Tasks().create_function(
            run_model, name=args.product_name, **DL_SYSTEM_PARAMS)

        for dlkey in tqdm(tiles):
            async_func(dlkey, **vars(args))

if __name__ == "__main__":
    main()
