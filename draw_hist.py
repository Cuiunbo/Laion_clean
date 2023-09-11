import argparse
import os
import dask
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import time
from glob import glob
from dask.distributed import Client
from dask import delayed
from datetime import datetime
import dask.array as da

if __name__ == '__main__':
    
    client = Client(n_workers=4, threads_per_worker=1)
    dask.config.set({"array.chunk-size": "5GB"})
    
    directory = '/mnt/alluxio/alluxio-fuse/user/tc_agi/klara/datasets/laion2b_en/laion2b_en_20230417112304'
#     max_width = 1200  # 宽度最大值
#     max_height = 1200  # 高度最大值
    bin_size = 50  # 直方图的 bin 大小
    
    start_time = time.time()
    
    files = glob(os.path.join(directory, '*.parquet'))
    files = [f for f in files if not os.path.basename(f).startswith('.')]
    # 扫描一部分数据以确定 max_width 和 max_height
    sample_files = files[:10] 
    sample_df = dd.read_parquet(sample_files)[['WIDTH', 'HEIGHT']]
    
    max_width = sample_df['WIDTH'].max().compute()
    max_height = sample_df['HEIGHT'].max().compute()
    total_hist, xedges, yedges = np.histogram2d([], [], bins=[np.arange(0, np.max(sample_df['WIDTH'].compute()), bin_size), np.arange(0, np.max(sample_df['HEIGHT'].compute()), bin_size)])
    print(f'Total files: {len(files)}, max_WxH: {max_width} x {max_height}')
    
    
    @dask.delayed
    def process_file(file):
        df = dd.read_parquet(file)[['WIDTH', 'HEIGHT']]
        df = df[(df['WIDTH'] < max_width) & (df['HEIGHT'] < max_height)]
        hist, _, _ = np.histogram2d(df['WIDTH'].compute(), df['HEIGHT'].compute(), bins=[xedges, yedges])
#         data = da.from_array(df.compute())
#         hist, edges = da.histogramdd(data, bins=[xedges, yedges])
        print(f'Processing file: {file} at {datetime.now()}')
#         return hist.compute()
        
        return hist
    
    hists = [process_file(file) for file in files]
    total_hist = sum(dask.compute(*hists))
    
    plt.imshow(total_hist, origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.xlabel('WIDTH')
    plt.ylabel('HEIGHT')
    plt.title('WIDTH and HEIGHT Distribution')
    plt.savefig('laion2b_histogram.png')
    np.save('laion2b_hist.npy', total_hist)
    np.save('laion2b_xedges.npy', xedges)
    np.save('laion2b_yedges.npy', yedges)
    plt.show()
    
    print(f'Total time: {time.time() - start_time} seconds.')
    
    client.close()

