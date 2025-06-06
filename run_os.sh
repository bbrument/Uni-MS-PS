#!/bin/bash

views=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 16 17 18 19 20 21 22 23)
for view in "${views[@]}"; do
    echo "Processing view $view"
    pad_view=$(printf "%02d" $view)
    python inference_file.py --nb_img 36 --folder_save exp --path_obj data/MVPS_CROP/view_$pad_view --cuda
done
