# requirements
```bash
pip install opencc dask faiss-gpu
pip install "dask[distributed]" --upgrade
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install -e .
```

# args

脚本参数说明：

- directory: 
    指定包含 parquet 文件的目录路径。

- crop_ratio_range:
    指定图像的可接受裁剪比例范围。
    默认值为 [0.8, 1.25]。

- image_size_range:
    指定图像的可接受大小范围。
    默认值为 [128, 1024]。

- text_length_range:
    指定文本的可接受长度范围。
    默认值为 [16, 512]。

- max_text_img_score:
    文本与图像相似性的最大可接受阈值。
    默认值为 0.6。

- max_img_img_score:
    图像与图像相似性的判断重复阈值。
    默认值为 0.98。

- num_neighbors (k):
    在img_img相似度矩阵中检索的最近邻居的数量。
    默认值为 20。

- retrieval_mode:
    数据检索模式。可以是 'full' 或 'partial'。
    默认值为 'partial'。

- export_discarded:
    如果设置，将导出被丢弃的数据。

# help
- 使用test_fina.ipynb获得更好的调试体验
- run.sh -> build.py

# pipline
laion_en -> 13W个parquet

下面是300个parquet的结果 (100个cpu 1000G内存, a800*1)

1. 文本和图像尺寸的清洗与过滤 -> 6min
	1. 图片 :长宽比, 长宽限制; 文本 : 字符转换, 语言清洗, 长度限制
	2. 这步将保留91w的数据
2. 图像计算clip特征 -> 100min
	1. 已经使用的 VIT-B-16了, 应该是相对小的 
	2. 锁线程, 保证GPU同时只给了唯一一个dataframe的partition
3. 计算特征相似度矩阵 -> 
	1. 未使用torch.cos, 而是手写, 据说更快, 因为用faiss
	2. 使用faiss计算邻居, 已知最快方法
