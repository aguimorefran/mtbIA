import os
import pandas as pd
from tcxreader.tcxreader import TCXReader
import tqdm

def process_tcx_files(read_folder_path="tcx", write_folder_path="csv"):
    files_tcx = [f for f in os.listdir(read_folder_path) if f.endswith('.tcx')]
    print("Number of TCX files:", len(files_tcx))

    reader = TCXReader()
    for file_tcx in tqdm.tqdm(files_tcx):
        data = reader.read(read_folder_path + "/" + file_tcx)
        data = data.trackpoints_to_dict()
        df = pd.DataFrame(data)
        df.to_csv(write_folder_path + "/" + file_tcx[:-4] + ".csv", index=False)
