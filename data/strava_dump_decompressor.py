import gzip
import os
import shutil

import tqdm

STRAVA_FOLDER = "strava_export"


# Decompress al .gz files in the strava_export/decompressed folder
# If decompressed folder does not exist, create it. If it exists, delete it and create it again recursively

def decompress_files():
    """
    Decompress all .gz files in the strava_export folder
    :return: None
    """
    gz_files = [f for f in os.listdir(STRAVA_FOLDER) if f.endswith('.gz')]
    print("Number of gz files:", len(gz_files))

    # Create the decompressed folder if it does not exist. If exists, delete it and create it again
    if os.path.exists(STRAVA_FOLDER + "/decompressed"):
        shutil.rmtree(STRAVA_FOLDER + "/decompressed")
    os.mkdir(STRAVA_FOLDER + "/decompressed")

    print("Decompressing files...")
    for gz_file in tqdm.tqdm(gz_files):
        with gzip.open(STRAVA_FOLDER + "/" + gz_file, 'rb') as f_in:
            with open(STRAVA_FOLDER + "/decompressed/" + gz_file[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    print("Done decompressing files")


decompress_files()
