#!/bin/bash

# run_script.sh

python build.py \
    # --directory "/path/to/your/directory" \
    --crop_ratio_range 0.8 1.25 \
    --image_size_range 128 1024 \
    --text_length_range 16 512 \
    --max_text_img_score 0.6 \
    --max_img_img_score 0.98 \
    --num_neighbors 20 \
    --retrieval_mode "partial" \
    # --export_discarded
