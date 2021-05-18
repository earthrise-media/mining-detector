import os

for region in range(1,7):
    print("Starting job for Bolivar region", region)
    os.system(f"python3 nn_output_fc.py --roi_file ../data/bolivar_{region}.geojson --tilesize 840 --product_id '28_px_bolivar-{region}_v9-2016' --product_name 'Bolivar part {region} - 28px Model V9 - 2016' --model_file ../models/28_px_v9.h5 --model_name '28_px_v9' --pad 0 --create_product --year 2016")
