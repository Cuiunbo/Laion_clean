import argparse
import os
import re
import io
import sys
import threading
import logging
import time
import datetime
from glob import glob
import pandas as pd
import numpy as np
import pyarrow as pa


import torch
import clip
import faiss
import dask
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client
from dask import delayed
from datetime import datetime
from PIL import Image


from utils.CleanData import Operate, filter_language
from utils.FilterData import FilterPara

# 设置日志级别
logging.basicConfig(level=logging.INFO)

start_time = time.time()


def ensure_directories_exist(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

class ImageTextProcessor:

    def __init__(self, args):
        
        self.directory = args.directory
        self.crop_ratio_range = args.crop_ratio_range
        self.image_size_range = args.image_size_range
        self.text_length_range = args.text_length_range
        self.export_discarded = args.export_discarded
        self.retrieval_mode = args.retrieval_mode
        self.max_text_img_score = args.max_text_img_score
        self.k = args.num_neighbors
        self.max_img_img_score = args.max_img_img_score
                
        self.files = None
        self.df = None
        self.filtered_df = None
        self.final_df = None

    def read_data(self):
        self.files = glob(os.path.join(self.directory, '*.parquet'))
        self.files = [f for f in self.files if not os.path.basename(f).startswith('.')]
        if self.retrieval_mode == 'partial':
            sample_files = self.files[:2]
        elif self.retrieval_mode == 'all':
            sample_files = self.files
            

        self.df = dd.read_parquet(sample_files)
        

    def filter_img(self):
        min_ratio, max_ratio = self.crop_ratio_range
        self.df['RATIO'] = self.df['WIDTH'] / self.df['HEIGHT']
        self.filtered_df = self.df[(self.df['RATIO'] >= min_ratio) & (self.df['RATIO'] <= max_ratio)]
        
        self.filtered_df = self.filtered_df.drop('RATIO', axis=1)


        min_width, max_width = self.image_size_range
        min_height, max_height = self.image_size_range

        self.filtered_df = self.filtered_df[(self.filtered_df['WIDTH'] < max_width) & (self.filtered_df['HEIGHT'] < max_height)]
        self.filtered_df = self.filtered_df[(self.filtered_df['WIDTH'] > min_width) & (self.filtered_df['HEIGHT'] > min_height)]

        # filter by similarity score
        self.filtered_df = self.filtered_df[(self.filtered_df['similarity'] <  self.max_text_img_score)]


    
    def process_text(self):
        min_length, max_length = self.text_length_range
        self.filtered_df = self.filtered_df[(self.filtered_df['TEXT'].str.len() >= min_length) & (self.filtered_df['TEXT'].str.len() <= max_length)]

        self.filtered_df['TEXT'] = self.filtered_df['TEXT'].map(Operate, meta=('TEXT', 'object'))
        self.filtered_df = self.filtered_df[self.filtered_df['TEXT'].apply(filter_language, meta=('TEXT', 'bool'))]

    def clip_score(self):
        desired_rows_per_partition = 6000
        batch_size = 3200
        df_clip = self.filtered_df.compute()
        npartitions = len(df_clip) // desired_rows_per_partition
        df_clip = dd.from_pandas(df_clip, npartitions=npartitions)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("/mnt/data/user/tc_agi/multi_modal/checkpoints/clip/ViT-B-16.pt", device=device)
        def process_image(row):
            try:
                img = Image.open(io.BytesIO(row['BUFFER']))
                tensor = preprocess(img).to(device)
                return tensor, row
            except Exception as e:
                print(f"Error processing image: {e}")
                return None, None

        gpu_lock = threading.Lock()

        def compute_embeddings(partition, batch_size=3200):
            with gpu_lock:

                all_embeddings = []
                valid_rows = []  

                for start in range(0, len(partition), batch_size):
                    end = start + batch_size
                    batch = partition.iloc[start:end]
                    print(len(batch))
                    processed_tensors = []
                    for _, row in batch.iterrows():
                        tensor, valid_row = process_image(row)
                        if tensor is not None:
                            processed_tensors.append(tensor)
                            valid_rows.append(valid_row)

                    tensor_stack = torch.stack(processed_tensors)
                    
                    with torch.no_grad():
                        embedding = model.encode_image(tensor_stack)
                    all_embeddings.extend(embedding.cpu().numpy())

                valid_partition = pd.concat(valid_rows, axis=1).transpose()
                valid_partition['CLIP_Features'] = all_embeddings

                return valid_partition

        df_clip = df_clip.map_partitions(compute_embeddings, batch_size=batch_size, meta=df_clip._meta.assign(CLIP_Features='f8'))
        df_clip['CLIP_Features'] = df_clip['CLIP_Features'].apply(
            lambda x: np.array(x).astype('float32') if isinstance(x, (list, np.ndarray)) else x,
            meta=('CLIP_Features', 'object')
        )

        self.final_df = df_clip

    def cal_img_img_score(self):
        k = self.k  
        threshold = self.max_img_img_score 

        df_pandas = self.final_df.compute()
        df_pandas = df_pandas.set_index('SAMPLE_ID')

        embeddings_matrix = np.vstack(df_pandas['CLIP_Features'].to_list()).astype('float32')  

        faiss.normalize_L2(embeddings_matrix)

        index = faiss.IndexFlatL2(embeddings_matrix.shape[1])  
        index.add(embeddings_matrix)

        D, I = index.search(embeddings_matrix, k)

        similar_pairs = []
        marked_items = set() 

        for i in range(I.shape[0]):
            if i in marked_items:
                continue

            similarities = 1 - D[i] / 2  
            potential_neighbors = I[i][similarities > threshold]
            
            filtered_sample_ids = [idx for idx in potential_neighbors if idx != i and idx not in marked_items]
            
            if filtered_sample_ids:
                similar_pairs.append((df_pandas.index[i], df_pandas.index[filtered_sample_ids].tolist()))
                marked_items.add(i)
                marked_items.update(filtered_sample_ids)
    

        return similar_pairs
    
    def deduplication(self,similar_pairs):
        def do():
            # method to deduplicate
            pass
        df = self.final_df
        df= df.map_partitions(do)
        df = df.drop('CLIP_Features', axis=1)
        self.final_df = df
        
        

    def run(self):
        self.read_data()
        self.filter_img()
        self.process_text()
        self.clip_score()
        self.deduplication(self.cal_img_img_score())
        
#         self.filtered_df = self.filtered_df.repartition(partition_size='500MB')


        schema = pa.schema([
            ('SAMPLE_ID', pa.float64()),
            ('URL', pa.string()),
            ('TEXT', pa.string()),
            ('HEIGHT', pa.int64()),
            ('WIDTH', pa.int64()),
            ('LICENSE', pa.string()),
            ('NSFW', pa.string()),
            ('similarity', pa.float64()),
            ('BUFFER', pa.binary()),  # 注意这里是 binary
            ('IMG_TYPE', pa.string()),
            # ('CLIP_Features', pa.list_(pa.float32())),  # 注意这里是 list<item: halffloat>
            ('__null_dask_index__', pa.int64())
        ])
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'../output_data_{timestamp}'
        discarded_dir = f'../discarded_data_{timestamp}'

        ensure_directories_exist([output_dir, discarded_dir])


        self.final_df.to_parquet(output_dir, schema=schema)
        # self.final_df.to_parquet(output_dir)
#         logging.info(f'size after process:{os.path.getsize("./test.parquet")/1024/1024}MB')
        
        if self.export_discarded:
            self.discarded_df = self.df.drop(self.final_df.index)
            self.discarded_df.to_parquet(discarded_dir, schema=schema) 
            logging.info(f'export discarded_data at {time.time() - start_time}s')
         
        logging.info(f'Total time: {time.time() - start_time} seconds.')


def main():
    parser = argparse.ArgumentParser(description='Process some images and text.')
    parser.add_argument('--directory', type=str, required= False, help='Path to directory containing parquet files')
    parser.add_argument('--crop_ratio_range', type=float, nargs=2, default=[0.8, 1.25], help='Range of crop ratio for images')
    parser.add_argument('--image_size_range', type=int, nargs=2, default=[128, 1024], help='Range of image size')
    parser.add_argument('--text_length_range', type=int, nargs=2, default=[16, 512], help='Range of text length')
    parser.add_argument('--max_text_img_score', type=float, default=0.6, help='Maximum text-image score')
    parser.add_argument('--max_img_img_score', type=float, default=0.98, help='Maximum image-image score')
    parser.add_argument('--num_neighbors', type=int, default=20, help='Number of closest neighbors to retrieve in the similarity matrix')

    parser.add_argument('--retrieval_mode', default='partial', choices=['full', 'partial'], help='Mode for data retrieval: full or partial')
    parser.add_argument('--export_discarded', action='store_true', help='Export discarded data')

    args = parser.parse_args()
    if not args.directory:
        args.directory = '/mnt/alluxio/alluxio-fuse/user/tc_agi/klara/datasets/laion2b_en/laion2b_en_20230417112304'
    processor = ImageTextProcessor(args)
    processor.run()

if __name__ == "__main__":
#     client = Client(n_workers=128, threads_per_worker=1)
#     dask.config.set({"array.chunk-size": "5GB"})
    main()
