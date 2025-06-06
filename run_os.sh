#!/bin/bash

DATA_PATH="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/real_data/os_gourdan/"
OUTPUT_PATH="/home/bbrument/dev/EVAL_DLMV_DTU/EVAL/real_data/os_gourdan/output"
# DATA_PATH="/home/lilian/dev/EVAL_DLMV_DTU/EVAL/real_data/os_gourdan/"
# OUTPUT_PATH="/home/lilian/dev/EVAL_DLMV_DTU/EVAL/real_data/os_gourdan/output"
mkdir -p $OUTPUT_PATH

# views=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 16 17 18 19 20 21 22 23)
views=(23 22 21 20 19 18 17 16 14 13 12 11 10 9 8 7 6 5 4 3 2 1)
for view in "${views[@]}"; do

    pad_ind=$(printf "%08d" $(($view - 1)))
    if [ ! -d "$DATA_PATH/view_${pad_ind}" ]; then
        echo "Directory for view $view does not exist, skipping."
        continue
    fi
    if [ -f "$OUTPUT_PATH/${pad_ind}.png" ]; then
        echo "Output file for view $view already exists, skipping."
        continue
    fi

    pad_view=$(printf "%02d" $view)
    if [ -f "$OUTPUT_PATH/view_${pad_view}.png" ]; then
        echo "Temporary file for view $view already exists, skipping."
        mv "$OUTPUT_PATH/view_${pad_view}.png" "$OUTPUT_PATH/${pad_ind}.png"
        continue
    fi

    echo "Processing view $view"
    python inference_file.py --nb_img 36 \
        --folder_save "$OUTPUT_PATH" \
        --path_obj "$DATA_PATH/view_${pad_view}" \
        --cuda

    mv "$OUTPUT_PATH/view_${pad_view}.png" "$OUTPUT_PATH/${pad_ind}.png"
done
