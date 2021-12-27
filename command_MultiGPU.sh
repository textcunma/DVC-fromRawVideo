#!/bin/bash
cd ./BMT/submodules/video_features/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate i3d
echo "I3D"
python main.py --feature_type i3d --on_extraction save_numpy --device_ids 0,1 --extraction_fps 25 --video_paths $1  --output_path $2      #extraction_fpsは、元動画の総フレーム数の半分の値をいれると元モデルと同じになるはず
conda deactivate
echo "I3D Finish"
echo "VGGish"
conda activate vggish
python main.py --feature_type vggish --on_extraction save_numpy --device_ids 0,1 --video_paths $1  --output_path $2
conda deactivate

