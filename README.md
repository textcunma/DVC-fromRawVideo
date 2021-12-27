# DVC-fromRawVideo
Dense Video Captioning from Raw Video（mp4動画のキャプション生成）/ 特徴抽出&amp;キャプション生成

- I3D PWC-Net VGGishの特徴を抽出
- BMTを用いてキャプションを生成

## 動作環境
- OS : Ubuntu 16.04 , Ubuntu 20.04
- GPU: GTX 1080Ti  ※「RTX3080Ti」では動作しません


## 環境構築
```bash
bash setup.sh
```

## 実行
inputsフォルダにmp4動画を入れた状態で実行. <br>
ファイル名は「main.py」で書き換えてください

```
def main():
    videoname="women_long_jump.mp4"  <---
```

## 注意
Miniconda ではなくAnacondaで実行したい場合はcommand_SingleGPU.shもしくはcommand_MultiGPU.shを書き換えてください
```bash
before: source ~/miniconda3/etc/profile.d/conda.sh
```
```bash
after: source ~/anaconda3/etc/profile.d/conda.sh
```

## エラー対応
239dimension error
1. add glove.840B.300.zip to ".vector_cashe(hidden directory)" https://nlp.stanford.edu/projects/glove/
2. And, open zip file