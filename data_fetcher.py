import os
import math
import requests
import pandas as pd
from time import sleep

print("Starting satellite image downloader (ESRI tiles)")

# paths
TRAIN_PATH = "data/raw/train(1).xlsx"
TEST_PATH = "data/raw/test2.xlsx"
IMAGE_DIR = "data/images"

os.makedirs(IMAGE_DIR, exist_ok=True)

# load data
train_df = pd.read_excel(TRAIN_PATH)
test_df = pd.read_excel(TEST_PATH)

all_data = pd.concat([train_df, test_df], ignore_index=True)

print("Total properties:", len(all_data))

# tile utils
def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
        / 2.0 * n
    )
    return xtile, ytile


def download_tile(x, y, z, save_path):
    url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True
    return False


# download 
ZOOM = 18
LIMIT = 1000   

print("Downloading test images...")

success = 0

for i, row in all_data.head(LIMIT).iterrows():
    prop_id = row["id"]
    lat = row["lat"]
    lon = row["long"]

    x, y = latlon_to_tile(lat, lon, ZOOM)
    img_path = os.path.join(IMAGE_DIR, f"{prop_id}.png")

    print(f"Property {prop_id} -> tile ({x}, {y})")

    ok = download_tile(x, y, ZOOM, img_path)

    if ok:
        print("Saved:", img_path)
        success += 1
    else:
        print("Failed:", prop_id)

    sleep(0.2)

print("Finished. Images saved:", success)
