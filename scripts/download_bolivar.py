import os

for region in range(1,7):
    print("Downloading predictions for Bolivar region", region)
    os.system(f"python3 download_dl_result.py --roi_file ../data/bolivar_{region}.geojson --product_id 28_px_bolivar-{region}_v9-2016")
