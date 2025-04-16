
## Animal Behavior Anomaly Detection via Spatio-Temporal Puzzle pre-train | AROB 2025 Experimental Repository |2025 AROB 実験リポジトリ 

### 🐾🐾🐾Overview  
This project focuses on the task of **anomaly detection in animal behavior videos** using the [Animal Kingdom dataset](https://sutdcv.github.io/Animal-Kingdom/).  <br>
For self-supervised pretraining, we adopt [Space-Time Jigsaw Puzzles](https://arxiv.org/pdf/1811.09795) and use it in [VAD](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_29).  <br>
The following models are supported:  
- [TimeSFormer](https://arxiv.org/pdf/2102.05095)  
- [SwinTransformer](https://arxiv.org/abs/2106.13230)  
- [C3D](https://arxiv.org/pdf/1412.0767)  
- [R3D](https://arxiv.org/pdf/1711.11248v3)
- 
### 🐾🐾🐾概要
動物行動動画[Animal Kingdomデータセット](https://sutdcv.github.io/Animal-Kingdom/)を対象とした異常行動検出タスクです。<br>
事前学習として[Space-Time Cubic Puzzles](https://arxiv.org/pdf/1811.09795)お使い、[VAD](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_29)を学習させる。<br>
対応しているモデルは下記の通り。
- [TimeSFormer](https://arxiv.org/pdf/2102.05095)
- [SwinTransformer](https://arxiv.org/abs/2106.13230)
- [C3D](https://arxiv.org/pdf/1412.0767)
- [R3D](https://arxiv.org/pdf/1711.11248v3)

### 🚀🚀🚀Setup  
Please follow the steps below to set up the environment before running the project.

1. To set up the environment directly using Conda:  <br>
    Run the following commands in sequence:
    ```
    conda create -n animal-VAD python=3.10 -y
    conda activate animal-VAD
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

### 🚀🚀🚀 セットアップ
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
        
### 🛠️🛠️🛠️ How to Use  
After setting up the environment, you can start training with the following command. Hyperparameters are specified in `config.yml`.  <br>
```
python main.py -config_path config.yml
```
In `config.yml`, specify the <u>model type</u>, <u>dataset type</u>, and training parameters:  
```
pretrain:
    is_valid    : False     # Enable/disable pretraining
    epoch       : 15        # Pretraining epochs
    batch_size  : 8         
    num_workers : 4
    optimizer   : adamw     
    lr          : 0.0005    
    weight_decay: 0.05      

train:
    epoch       : 30        # Training epochs
    batch_size  : 8         
    num_workers : 4
    optimizer   : adamw     
    lr          : 0.0005    
    weight_decay: 0.05      

dataset:
    type: K400-S-Separate   # Dataset type

model:
    type: TimeSFormer       # Model type

others:
    rnd_seed      : 1
    running_path  : ./logs
    result_path   : ./results
```

- Experiment Shell Script  <br>
  If you want to run multiple experiments, save each config in the `configs` folder and execute `run.sh`.  <br>
  The script will run `main.py` for each `.yml` file in the folder automatically.  
```
./run.sh
```

- Model Type  <br>
Model types are defined in `model/model_zoo.yml`:  
```
# Example: VideoSwinTransformer base model definition
Swin-B:
    Architecture: SwinTransformer3D
    patch_size: (4,4,4)
    embed_dim: 128
    depths: [2, 2, 18, 2]
    num_heads: [4,8,16,32]
    window_size: (8,7,7)
    mlp_ratio: 4.
    qkv_bias: True
    qk_scale: None
    drop_rate: 0.
    attn_drop_rate: 0.
    drop_path_rate: 0.2
    patch_norm: True

    cls_head:
        in_channels: 1024
```

- Dataset Type  <br>
Dataset types are defined in `dataset/dataset_zoo.yml`. The `pretrain_type` is defined in `dataset/pretrain_scheme/pretrain_zoo.yml`.  
```
☆☆☆ dataset_zoo.yml ☆☆☆
# Example: KineticsK400 dataset definition
K400-S-Separate:
    datatype          : KineticsK400
    num_classes       : 400
    n_sample_per_class: 0.3   # Use 30% of data per class for classification

    img_size          : 224
    n_frames          : 16
    frame_interval    : 2
    
    train_path        : /mnt/d/Work/Create/RD_REPOS/20240808_Transformer系のモデルに3DCubicPuzzleを応用する/K400_train
    valid_path        : /mnt/d/Work/Create/RD_REPOS/20240808_Transformer系のモデルに3DCubicPuzzleを応用する/K400_valid
    movie_ext         : .mp4

    train_transforms: 
        scale       : None
        ratio       : None
        hflip       : 0.5
        color_jitter: 0.4
        norm_mean   : (0.485, 0.456, 0.406)
        norm_std    : (0.229, 0.224, 0.225)

    valid_transforms: 
        norm_mean   : (0.485, 0.456, 0.406)
        norm_std    : (0.229, 0.224, 0.225)

    pretrain:
        pretrain_type     : SeparateJigsawPuzzle3D
        pretrain_path     : /mnt/d/Work/Create/RD_REPOS/20240808_Transformer系のモデルに3DCubicPuzzleを応用する/K400_train
        n_sample_per_class: 1.0
```

```
☆☆☆ pretrain_zoo.yml ☆☆☆
# Example: JointJigsawPuzzle3D definition
JointJigsawPuzzle3D:
    type             : JointJigsawPuzzle3D
    n_grid           : (2,2,4)         # Grid split in (h, w, t)
    mask_grid_ratio  : 0.125           # Masking ratio for pretraining
```

- Data Path Configuration  <br>
Specify `train_path`, `valid_path`, and `pretrain_path` in `dataset_zoo.yml`. The dataset folder should follow the structure below:  
```
train_path
├── classmap.json       # JSON mapping of class names to indices
│                       # Example: { "CLASS_A":1, "CLASS_B":2, "CLASS_C":3 }
├── CLASS_A
│   ├── 001.mp4
│   ├── 002.mp4
│   └── 003.mp4
├── CLASS_B
│   ├── 001.mp4
│   ├── 002.mp4
│   └── 003.mp4
└── CLASS_C
    ├── 001.mp4
    ├── 002.mp4
    └── 003.mp4
```
### 🛠️🛠️🛠️ 使い方
- セットアップで環境構築した後で下記でトレーニングできます。学習のハイパーパラメータはconfig.ymlで指定できます。
    ```
    python main.py -config_path config.yml
    ```
    config.ymlでは使用する<u>モデルタイプ</u>と<u>データセットタイプ</u>、学習のパラメータを指定します。
    ```
    pretrain:
        is_valid    : False     # 事前学習の有無(Falseは無)
        epoch       : 15        # 事前学習エポック数
        batch_size  : 8         # バッチ数
        num_workers : 4
        optimizer   : adamw     # オプティマイザ
        lr          : 0.0005    # 学習率
        weight_decay: 0.05      # 重み正則化

    train:
        epoch       : 30        # トレーニングエポック数
        batch_size  : 8         # バッチ数
        num_workers : 4
        optimizer   : adamw     # オプティマイザ
        lr          : 0.0005    # 学習率
        weight_decay: 0.05      # 重み正則化

    dataset:
        type: K400-S-Separate #データセットタイプは「K400-S-Separate」を使用

    model:
        type: TimeSFormer    #モデルタイプは「TimeSFormer」を使用

    others:
        rnd_seed      : 1
        running_path  : ./logs
        result_path   : ./results
    ```
- 実験用シェルスクリプト<br>
 複数の実験設定の下で比較実験を行いたい場合、実験設定をconfig.ymlに記述し、順次configsフォルダへ格納していき、「run.sh」を実行します。<br>
 「run.sh」はconfigsフォルダ内のymlファイルを引数として、「main.py」を順次実行するため、自動で実験が進んでいきます。
    ```
    ./run.sh
    ```

- モデルタイプ<br>
モデルタイプは下記のようにmodel/model_zoo.ymlで定義します。
    ```
    # VideoSwinTransformerのベースモデルの定義例
    Swin-B:
        Architecture: SwinTransformer3D
        patch_size: (4,4,4)
        embed_dim: 128
        depths: [2, 2, 18, 2]
        num_heads: [4,8,16,32]
        window_size: (8,7,7)
        mlp_ratio: 4.
        qkv_bias: True
        qk_scale: None
        drop_rate: 0.
        attn_drop_rate: 0.
        drop_path_rate: 0.2
        patch_norm: True

        cls_head:
            in_channels: 1024
    ```
- データセットタイプ<br>
データセットタイプは下記のようにdataset/dataset_zoo.ymlで定義します。なお、事前学習のタイプ「pretrain_type」はdataset/pretrain_scheme/pretrain_zoo.ymlで定義しておきます。
    ```
    ☆☆☆　dataset_zoo.yml　☆☆☆
    # KineticsK400のデータセット定義例
    K400-S-Separate:
        datatype          : KineticsK400
        num_classes       : 400
        n_sample_per_class: 0.3   # クラス分類時に使うデータセットは全体の30%

        img_size          : 224
        n_frames          : 16
        frame_interval    : 2
        
        train_path        : /mnt/d/Work/Create/RD_REPOS/20240808_Transformer系のモデルに3DCubicPuzzleを応用する/K400_train # トレーニングデータへのパス
        valid_path        : /mnt/d/Work/Create/RD_REPOS/20240808_Transformer系のモデルに3DCubicPuzzleを応用する/K400_valid # 検証データへのパス
        movie_ext         : .mp4

        train_transforms: 
            scale       : None
            ratio       : None
            hflip       : 0.5
            color_jitter: 0.4
            norm_mean   : (0.485, 0.456, 0.406)
            norm_std    : (0.229, 0.224, 0.225)

        valid_transforms: 
            norm_mean   : (0.485, 0.456, 0.406)
            norm_std    : (0.229, 0.224, 0.225)

          pretrain:
            pretrain_type     : SeparateJigsawPuzzle3D # 事前学習のタイプは「SeparateJigsawPuzzle3D」
            pretrain_path     : /mnt/d/Work/Create/RD_REPOS/20240808_Transformer系のモデルに3DCubicPuzzleを応用する/K400_train
            n_sample_per_class: 1.0 # 事前学習時に使うデータセットは100%
    ```
    ```
    ☆☆☆　pretrain_zoo.yml　☆☆☆
    # JointJigsawPuzzle3Dの定義例
    JointJigsawPuzzle3D:
        type          : JointJigsawPuzzle3D # Jigsawのタイプは「Joint」
        n_grid        : (2,2,4) # (h,w,t)の次元で分割するグリッド数は(2,2,4)
        mask_grid_ratio: 0.125  # マスクするグリッドの割合
    ```

- 使用するデータセットへのパスはdataset_zoo.ymlのtrain_pathおよびvalid_path、pretrain_pathに記述してください。<br>
  なお、データフォルダ内の構造は以下のような構造になっている必要があります。<br>
  ```
  train_path
    ├── classmap.json：データセットのクラス名と対応するインデックスを記述したjsonファイル。
    |                　下記の例では{ "CLASS_A":1,"CLASS_B":2,"CLASS_C":3}となる。
    ├── CLASS_A
    │   ├── 001.mp4：クラスAに属する学習データ
    │   ├── 002.mp4
    │   └── 003.mp4
    ├── CLASS_B
    │   ├── 001.mp4：クラスBに属する学習データ
    │   ├── 002.mp4
    │   └── 003.mp4
    └── CLASS_C
        ├── 001.mp4：クラスCに属する学習データ
        ├── 002.mp4
        └── 003.mp4
  ```
