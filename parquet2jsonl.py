import pandas as pd
import json
import os
import random
import time
import numpy as np


PARQUET_FILE_DIR = './data/starcoderdata/'
JSON_FILE_DIR = './data/starcoderdata_jsonl/'
# change the key name from 'content' to 'text'
USE_TEXT = True


FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"
FIM_PAD = "<fim_pad>"

# fix seed
random.seed(0)
np.random.seed(0)


# collect all parquet file path and json file path
parquet_file_dir = PARQUET_FILE_DIR
parquet_files = []
for file in os.listdir(parquet_file_dir):
    subdir = parquet_file_dir + file
    if os.path.isdir(subdir):
        for subfile in os.listdir(subdir):
            if subfile.endswith('.parquet'):
                parquet_files.append(subdir + '/' + subfile)

def get_json_file_paths(dir):
    json_file_dir = dir
    json_files = []
    if not os.path.exists(json_file_dir):
        os.makedirs(json_file_dir)
    for file in parquet_files:
        json_file_subdir = json_file_dir + file.split('/')[-2] + '/'
        if not os.path.exists(json_file_subdir):
            os.makedirs(json_file_subdir)
        json_files.append(json_file_subdir + file.split('/')[-1].replace('.parquet', '.jsonl'))
    return json_files

json_files = get_json_file_paths(JSON_FILE_DIR)

def parquet_to_json(parquet_file, json_file):
    df = pd.read_parquet(parquet_file)
    if USE_TEXT:
        df.rename(columns={'content': 'text'}, inplace=True)

    # write original jsonl file
    with open(json_file, 'a') as f:
        df.to_json(f, orient='records', lines=True)



begin_time = time.time()
total_num = len(parquet_files)
try:
    for i, (parquet_file, json_file) in enumerate(zip(parquet_files, json_files)):
        parquet_to_json(parquet_file, json_file)

        if i % 5 == 0:
            print('finished {}/{} files, accumulated time: {} minutes, current process: {}'.format(i, total_num, round((time.time() - begin_time) / 60, 2), parquet_file.split('/')[-2]))
except:
    print('error in {}'.format(parquet_file))            