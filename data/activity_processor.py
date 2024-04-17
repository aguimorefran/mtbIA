import os

import pandas as pd
import tqdm
import shutil
from tcxreader.tcxreader import TCXReader
import fitparse


def process_tcx_files(read_folder_path, write_folder_path):
    """
    Process all TCX files in a folder and save them as CSV files
    :param read_folder_path:  The folder where the TCX files are located
    :param write_folder_path: The folder where the CSV files will be saved
    :return: True if the process was successful, False otherwise
    """
    files_tcx = [f for f in os.listdir(read_folder_path) if f.endswith('.tcx')]
    print("-----------------------------------")
    print("Processing TCX files in", read_folder_path)
    print("Number of TCX files:", len(files_tcx))

    reader = TCXReader()
    try:
        for file_tcx in tqdm.tqdm(files_tcx):
            data = reader.read(read_folder_path + "/" + file_tcx)
            data = data.trackpoints_to_dict()
            df = pd.DataFrame(data)
            df.to_csv(write_folder_path + "/" + file_tcx[:-4] + ".csv", index=False)
    except Exception as e:
        print("Error processing TCX files:", e)
        return False
    print("Done processing TCX files")

    return True



def process_fit_files(read_folder_path, write_folder_path):
    """
    Process all FIT files in a folder and save them as CSV files
    :param read_folder_path:  The folder where the FIT files are located
    :param write_folder_path: The folder where the CSV files will be saved
    :return: True if the process was successful, False otherwise
    """
    files_fit = [f for f in os.listdir(read_folder_path) if f.endswith('.fit')]
    print("-----------------------------------")
    print("Processing FIT files in", read_folder_path)
    print("Number of FIT files:", len(files_fit))

    try:
        for file_fit in tqdm.tqdm(files_fit):
            with fitparse.FitFile(read_folder_path + "/" + file_fit) as fitfile:
                records = []
                for record in fitfile.get_messages('record'):
                    d = {}
                    for data in record:
                        d[data.name] = data.value
                    records.append(d)

                df = pd.DataFrame(records)
                df.to_csv(write_folder_path + "/" + file_fit[:-4] + ".csv", index=False)

    except Exception as e:
        print("Error processing FIT files:", e)
        return False

def process(read_folder_path, write_folder_path):
    if os.path.exists(write_folder_path):
        shutil.rmtree(write_folder_path)
    os.makedirs(write_folder_path)

    source_folders = [
        "strava_export/decompressed",
        "tcx_kaggle"
    ]

    destination_folder = "processed_activities"

    for source_folder in source_folders:
        process_tcx_files(source_folder, destination_folder)

    for source_folder in source_folders:
        process_fit_files(source_folder, destination_folder)


process("strava_export/decompressed", "processed_activities")