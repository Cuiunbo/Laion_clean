from utils.CleanData import Operate, filter_language
from utils.FilterData import FilterPara

import argparse
import os
import re
import time
import string
import io
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
import torch
import clip
import dask
import dask.array as da

from datetime import datetime
from PIL import Image
from glob import glob
from dask.distributed import Client
from dask import delayed

# client = Client(n_workers=128, threads_per_worker=1)
dask.config.set({"array.chunk-size": "5GB"})

directory = '/mnt/alluxio/alluxio-fuse/user/tc_agi/klara/datasets/laion2b_en/laion2b_en_20230417112304'
files = glob(os.path.join(directory, '*.parquet'))
files = [f for f in files if not os.path.basename(f).startswith('.')]
print("-----------------------------------")
sample_files = files[100:110]
sample_df = dd.read_parquet(sample_files)

min_ratio = 0.9
max_ratio = 1.1

sample_df['RATIO'] = sample_df['WIDTH'] / sample_df['HEIGHT']

filtered_df = sample_df[(sample_df['RATIO'] >= min_ratio) & (sample_df['RATIO'] <= max_ratio)]
filtered_df = filtered_df.drop('RATIO', axis=1)

min_width, max_width = 128, 1024
min_height, max_height = 128, 1024

filtered_df = filtered_df[(filtered_df['WIDTH'] < max_width) & (filtered_df['HEIGHT'] < max_height)]
filtered_df = filtered_df[(filtered_df['WIDTH'] > min_width) & (filtered_df['HEIGHT'] > min_height)]

min_length, max_length = 16,1024
df_len = filtered_df
df_len= df_len[(df_len['TEXT'].str.len() >= min_length) & (df_len['TEXT'].str.len() <= max_length)]

df = df_len
df['TEXT'] = df['TEXT'].map(Operate, meta=('TEXT', 'object'))
filtered_lang_df = df[df['TEXT'].apply(filter_language, meta=('TEXT', 'bool'))]

# 
start_time = time.time()
filtered_lang_df.compute()
print(f"1. {time.time() - start_time}")
# 

# time load img
start_time = time.time()
df_check = filtered_lang_df
def check_image_integrity(data):
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # 验证图像的完整性
        return False
    except OSError:
        return True

def check_images_integrity(partition):
    # partition['Corrupted'] = partition['BUFFER'].apply(check_image_integrity)
    partition.loc[:, 'Corrupted'] = partition['BUFFER'].apply(check_image_integrity)

    return partition

meta = df_check._meta.assign(Corrupted=pd.Series([False], dtype=bool))
df_check = df_check.map_partitions(check_images_integrity, meta=meta)

corrupted_df = df_check[df_check['Corrupted'] == True].compute()
print(f"2. {time.time() - start_time}")

# time load clip
df_clip = filtered_lang_df
# df = filtered_df
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/mnt/data/user/tc_agi/multi_modal/checkpoints/clip/ViT-B-16.pt", device=device)
print("-------------------------")

start_time = time.time()
# time embedding
def safe_open(data):
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  
        return img
    except Exception:
        return None

def compute_embeddings_with_exceptions(partition, batch_size=4000):
    all_embeddings = []

    partition['Image'] = partition['BUFFER'].apply(safe_open)
    partition['Is_Exception'] = partition['Image'].isnull()
    valid_images = partition.loc[~partition['Is_Exception'], 'Image'].tolist()

    for start in range(0, len(valid_images), batch_size):
        end = start + batch_size
        batch_images = valid_images[start:end]

        processed_images = []
        for img in batch_images:
            # try:
            processed_img = preprocess(img)  
            processed_images.append(processed_img)
            # except Exception:
                # 如果在预处理阶段发生异常，可以在这里处理，但对于您的用例，可能不需要
                # print("pass")
                # pass
        if processed_images:
            tensor = torch.stack(processed_images).to(device)
            with torch.no_grad():
                embedding = model.encode_image(tensor)
            all_embeddings.extend(embedding.cpu().numpy())

    partition['CLIP_Features'] = all_embeddings[:len(partition)]
    
    partition = partition.drop(columns=['Image'])
    return partition

# time
meta = df_clip._meta.assign(CLIP_Features=pd.Series(dtype='object'))
df_clip = df_clip.map_partitions(compute_embeddings_with_exceptions, meta=meta).compute()

print(f"3. {time.time() - start_time}")

# print
all_exceptions = df_clip[df_clip['Is_Exception']].index
# print

start_time = time.time()

df_pandas = df_clip
df_pandas = df_pandas.set_index('SAMPLE_ID')

# time
import faiss
import numpy as np

embeddings_matrix = np.vstack(df_pandas['CLIP_Features'].to_list()).astype('float32')  

# 如果embeddings未被归一化，请进行归一化
faiss.normalize_L2(embeddings_matrix)

index = faiss.IndexFlatL2(embeddings_matrix.shape[1])  
index.add(embeddings_matrix)

# 对每个embedding搜索其最近邻
D, I = index.search(embeddings_matrix, embeddings_matrix.shape[0])


threshold = 0.90  # 余弦相似度的阈值
similar_pairs = []

for i in range(I.shape[0]):
    similarities = 1 - D[i] / 2  # 将L2距离转换为余弦相似度
    filtered_sample_ids = df_pandas.index[I[i][(similarities > threshold) & (I[i] != i)]].tolist()
    if filtered_sample_ids:
        similar_pairs.append((df_pandas.index[i], filtered_sample_ids))

print(f"4. {time.time() - start_time}")

# print
