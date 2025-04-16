
## Animal Behavior Anomaly Detection via Spatio-Temporal Puzzle pre-train | AROB 2025 Experimental Repository |2025 AROB 実験リポジトリ 

### 🐾Overview  
This project focuses on the task of **anomaly detection in animal behavior videos** using the [Animal Kingdom dataset](https://sutdcv.github.io/Animal-Kingdom/).  <br>
For self-supervised pretraining, we adopt [Space-Time Jigsaw Puzzles](https://arxiv.org/pdf/1811.09795) and use it in [VAD](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_29).  <br>
The following models are supported:  
- [TimeSFormer](https://arxiv.org/pdf/2102.05095)  
- [SwinTransformer](https://arxiv.org/abs/2106.13230)  
- [C3D](https://arxiv.org/pdf/1412.0767)  
- [R3D](https://arxiv.org/pdf/1711.11248v3)
- 
### 🐾概要
動物行動動画[Animal Kingdomデータセット](https://sutdcv.github.io/Animal-Kingdom/)を対象とした異常行動検出タスクです。<br>
事前学習として[Space-Time Cubic Puzzles](https://arxiv.org/pdf/1811.09795)お使い、[VAD](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_29)を学習させる。<br>
対応しているモデルは下記の通り。
- [TimeSFormer](https://arxiv.org/pdf/2102.05095)
- [SwinTransformer](https://arxiv.org/abs/2106.13230)
- [C3D](https://arxiv.org/pdf/1412.0767)
- [R3D](https://arxiv.org/pdf/1711.11248v3)

### 🚀Setup  
Please follow the steps below to set up the environment before running the project.

1. To set up the environment directly using Conda:  <br>
    Run the following commands in sequence:
    ```
    conda create -n env-ViViT python=3.10 -y
    conda activate env-ViViT
    pip install --upgrade pip
    pip install -r requirements.txt
    conda install tensorboard
    pip install numpy==1.26.4 # Reinstall numpy because tensorboard may upgrade it
    ```

2. To set up the environment using Docker:  <br>
    Follow these steps:  <br>
    1. Build an image from the Dockerfile:
        ```
        docker build . -t img_env-ViViT
        ```
    2. Create a container from the image:
        ```
        docker run -v ./:/work -n env-ViViT -it img_env-ViViT
        ```

### 🚀セットアップ
実行する前に下記手順を参考にして環境構築してください。
1. Condaで直接環境構築する場合<br>
    下記コマンドを順番に実行して環境構築してください。
    ```
    conda create -n env-ViViT python=3.10 -y
    conda activate env-ViViT
    pip install --upgrade pip
    pip install -r requirements.txt
    conda install tensorboard
    pip install numpy==1.26.4 # tensorboardをインストールするとnumpyがアップグレードされるため再インストール
    ```
2. Dockerで環境構築する場合<br>
    下記手順で環境構築してください。<br>
    1. Dockerfileからイメージ構築
        ```
        docker build . -t img_env-ViViT
        ```
    2. イメージからコンテナ作成
        ```
        docker run -v ./:/work -n env-ViViT -it img_env-ViViT
        ```
