#!/bin/bash

#環境構築
#Dense Video Captioning with Bi-modal Transformer DOWNLOAD
git clone --recursive https://github.com/v-iashin/BMT.git
conda env create -f ./BMT/conda_env.yml
conda activate bmt
python -m spacy download en
conda deactivate

conda env create -f ./BMT/submodules/video_features/conda_env_i3d.yml
conda env create -f ./BMT/submodules/video_features/conda_env_vggish.yml
wget https://storage.googleapis.com/audioset/vggish_model.ckpt -P ./BMT/submodules/video_features/models/vggish/checkpoints

#BMTの学習済み重みをダウンロード
mkdir best_model
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/best_cap_model.pt -P ./best_model/
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/best_prop_model.pt -P ./best_model/