import os
import sys
import math

# カレントディレクトリをパスに追加
sys.path.append(os.getcwd())

# 必要なモジュールのインポート
from utils.utils import load_yml
from utils.transforms import getSpatialTransforms, getTemporalTransforms
from dataset.datasets.DatasetK400 import DatasetK400, PreTrainDatasetK400
from dataset.datasets.DatasetUCF101 import DatasetUCF101, PreTrainDatasetUCF101
from dataset.datasets.DatasetAnimalKingdom import DatasetAnimalKingdom, PreTrainAnimalKindom

# グローバル変数でデータセットの設定を保持
dataset_params = None

# データセット定義ファイルのパスを設定
yml_path = os.path.join(os.path.dirname(__file__), "dataset_zoo.yml")
dataset_zoo = load_yml(yml_path)  # データセット定義ファイルを読み込む


def initDatasetConfig(dataset_type: str):
    """
    データセットの設定を初期化する関数。
    指定されたデータセットタイプに基づき、グローバル変数 dataset_params に設定をロードする。

    Args:
        dataset_type (str): データセットの種類（例："K400-Base"）。

    Raises:
        Exception: データセットタイプがデータセット定義ファイルに存在しない場合。
    """
    if not hasattr(dataset_zoo, dataset_type):
        raise Exception(
            f"データセット定義ファイル({os.path.join(os.path.dirname(__file__), 'dataset_zoo.yml')})にタイプ{dataset_type}が定義されていません。"
        )
    global dataset_params
    dataset_params = dataset_zoo.__getattribute__(dataset_type)


    


def getDataset() -> dict:
    """
    指定されたデータセットタイプに基づき、学習用と検証用のデータセットを生成する。

    Returns:
        dict: 学習用および検証用のデータセットとクラス数の情報を含む辞書。

    Raises:
        Exception: データセットが存在しない場合。
    """
    if dataset_params.datatype.upper() == "KINETICSK400":
        # Kinetics400 データセットの生成
        print("データタイプ：KineticsK400")

        # 学習用データセットのパス確認とインスタンス作成
        DatasetK400.check_datapath(dataset_params.train_path, mode="train")
        train_transforms = getSpatialTransforms(
            img_size=dataset_params.img_size,
            scale=dataset_params.train_transforms.scale,
            ratio=dataset_params.train_transforms.ratio,
            hflip=dataset_params.train_transforms.hflip,
            color_jitter=dataset_params.train_transforms.color_jitter,
            mean=dataset_params.train_transforms.norm_mean,
            std=dataset_params.train_transforms.norm_std,
        )
        temporal_transforms = getTemporalTransforms(
            n_frames=dataset_params.n_frames,
            frame_interval=dataset_params.frame_interval,
        )
        train_dataset = DatasetK400(
            data_path=dataset_params.train_path,
            spatial_transform=train_transforms,
            temporal_transform=temporal_transforms,
            n_frames=dataset_params.n_frames,
            n_sample_per_class=dataset_params.n_sample_per_class,
            movie_ext=dataset_params.movie_ext,
        )

        # 検証用データセットのパス確認とインスタンス作成
        DatasetK400.check_datapath(dataset_params.valid_path, mode="valid")
        valid_transforms = getSpatialTransforms(
            img_size=dataset_params.img_size,
            mean=dataset_params.valid_transforms.norm_mean,
            std=dataset_params.valid_transforms.norm_std,
        )
        valid_dataset = DatasetK400(
            data_path=dataset_params.valid_path,
            spatial_transform=valid_transforms,
            temporal_transform=temporal_transforms,
            n_frames=dataset_params.n_frames,
            n_sample_per_class=dataset_params.n_sample_per_class,
            movie_ext=dataset_params.movie_ext,
        )

    elif dataset_params.datatype.upper() == "UCF101":
        # UCF101 データセットの生成
        print("データタイプ：UCF101")

        # 学習用データセットのパス確認とインスタンス作成
        DatasetUCF101.check_datapath(dataset_params.train_path, mode="train")
        train_transforms = getSpatialTransforms(
            img_size=dataset_params.img_size,
            scale=dataset_params.train_transforms.scale,
            ratio=dataset_params.train_transforms.ratio,
            hflip=dataset_params.train_transforms.hflip,
            color_jitter=dataset_params.train_transforms.color_jitter,
            mean=dataset_params.train_transforms.norm_mean,
            std=dataset_params.train_transforms.norm_std,
        )
        temporal_transforms = getTemporalTransforms(
            n_frames=dataset_params.n_frames,
            frame_interval=dataset_params.frame_interval,
        )
        train_dataset = DatasetUCF101(
            data_path=dataset_params.train_path,
            spatial_transform=train_transforms,
            temporal_transform=temporal_transforms,
            n_frames=dataset_params.n_frames,
            n_sample_per_class=dataset_params.n_sample_per_class,
            movie_ext=dataset_params.movie_ext,
        )

        # 検証用データセットのパス確認とインスタンス作成
        DatasetUCF101.check_datapath(dataset_params.valid_path, mode="valid")
        valid_transforms = getSpatialTransforms(
            img_size=dataset_params.img_size,
            mean=dataset_params.valid_transforms.norm_mean,
            std=dataset_params.valid_transforms.norm_std,
        )
        valid_dataset = DatasetUCF101(
            data_path=dataset_params.valid_path,
            spatial_transform=valid_transforms,
            temporal_transform=temporal_transforms,
            n_frames=dataset_params.n_frames,
            n_sample_per_class=1.0,
            movie_ext=dataset_params.movie_ext,
        )
    
    elif dataset_params.datatype.upper() == "ANIMALKINGDOM":
        # UCF101 データセットの生成
        print("データタイプ：animal_kingdom")

        # 学習用データセットのパス確認とインスタンス作成
        DatasetAnimalKingdom.check_datapath(dataset_params.train_path, mode="train")
        train_transforms = getSpatialTransforms(
            img_size=dataset_params.img_size,
            scale=dataset_params.train_transforms.scale,
            ratio=dataset_params.train_transforms.ratio,
            hflip=dataset_params.train_transforms.hflip,
            color_jitter=dataset_params.train_transforms.color_jitter,
            mean=dataset_params.train_transforms.norm_mean,
            std=dataset_params.train_transforms.norm_std,
        )
        temporal_transforms = getTemporalTransforms(
            n_frames=dataset_params.n_frames,
            frame_interval=dataset_params.frame_interval,
        )
        train_dataset = DatasetAnimalKingdom(
            data_path=dataset_params.train_path,
            spatial_transform=train_transforms,
            temporal_transform=temporal_transforms,
            n_frames=dataset_params.n_frames,
            n_sample_per_class=dataset_params.n_sample_per_class,
            movie_ext=dataset_params.movie_ext,
        )

        # 検証用データセットのパス確認とインスタンス作成
        DatasetAnimalKingdom.check_datapath(dataset_params.valid_path, mode="valid")
        valid_transforms = getSpatialTransforms(
            img_size=dataset_params.img_size,
            mean=dataset_params.valid_transforms.norm_mean,
            std=dataset_params.valid_transforms.norm_std,
        )
        valid_dataset = DatasetAnimalKingdom(
            data_path=dataset_params.valid_path,
            spatial_transform=valid_transforms,
            temporal_transform=temporal_transforms,
            n_frames=dataset_params.n_frames,
            n_sample_per_class=1.0,
            movie_ext=dataset_params.movie_ext,
        )    
    else:
         raise Exception(f"未対応のデータタイプ: {dataset_params.datatype}")

    return {"train_dataset": train_dataset, "valid_dataset": valid_dataset, "num_classes": dataset_params.num_classes}

from dataset.pretrain_scheme.pretrain_schemes.CubicPuzzle3D import CubicPuzzle3D
from dataset.pretrain_scheme.pretrain_schemes.SeparateJigsawPuzzle3D import SeparateJigsawPuzzle3D
from dataset.pretrain_scheme.pretrain_schemes.JointJigsawPuzzle3D import JointJigsawPuzzle3D

pretrain_scheme_params=None
pretrain_yml_path = os.path.join(os.path.dirname(__file__), "pretrain_scheme/pretrain_zoo.yml")
pretrain_scheme_zoo = load_yml(pretrain_yml_path)  # データセット定義ファイルを読み込む

def initPretrainScheme(pretrain_type: str):
    """
    指定された事前学習スキームの設定を初期化する関数。

    Args:
        pretrain_type (str): 事前学習スキームの種類（例: "CUBICPUZZLE3D", "SEPARATEJIGSAWPUZZLE3D", "JOINTJIGSAWPUZZLE3D"）。

    Raises:
        Exception: 指定された事前学習スキームが pretrain_zoo.yml に存在しない場合。
    """
    # 指定したスキームが定義されているか確認
    if not hasattr(pretrain_scheme_zoo, pretrain_type):
        raise Exception(
            f"事前学習定義ファイル({os.path.join(os.path.dirname(__file__), 'pretrain_scheme/pretrain_zoo.yml')})にスキーム {pretrain_type} が定義されていません。"
        )

    global pretrain_scheme_params
    # 指定したスキームの設定を取得し、グローバル変数に保存
    pretrain_scheme_params = pretrain_scheme_zoo.__getattribute__(pretrain_type)


def getPreTrainDataset() -> dict:
    """
    指定された事前学習スキームに基づき、事前学習用データセットを生成する関数。

    Returns:
        dict: 事前学習用データセットと空間および時間分類のクラス数を含む辞書。
              形式: {"pretrain_dataset": Dataset, "num_classes": (num_classes_spatial, num_classes_temporal)}。

    Raises:
        Exception: 必要なスキームやデータが不足している場合。
    """
    # 事前学習スキームを初期化
    initPretrainScheme(dataset_params.pretrain.pretrain_type)

    # スキームタイプに応じて処理を分岐
    if pretrain_scheme_params.type.upper() == "CUBICPUZZLE3D":
        print("事前学習タイプ：CubicPuzzle3D")
        # フレーム数をスキーム設定から計算
        n_frames = int(pretrain_scheme_params.jitter_size[2] * pretrain_scheme_params.n_grid[2])
        # CubicPuzzle3D のスキーム設定をインスタンス化
        pretrain_scheme = CubicPuzzle3D(
            img_size=dataset_params.img_size,
            n_frames=dataset_params.n_frames,
            n_grid=pretrain_scheme_params.n_grid,
            grayscale_prob=pretrain_scheme_params.grayscale_prob,
            jitter_size=pretrain_scheme_params.jitter_size,
            mean=dataset_params.train_transforms.norm_mean,
            std=dataset_params.train_transforms.norm_std,
        )
        # 空間・時間分類のクラス数を計算
        num_classes_spatial = math.factorial(int(pretrain_scheme_params.n_grid[0] * pretrain_scheme_params.n_grid[1]))
        num_classes_temporal = math.factorial(int(pretrain_scheme_params.n_grid[2]))

    elif pretrain_scheme_params.type.upper() == "SEPARATEJIGSAWPUZZLE3D":
        print("事前学習タイプ：SeparateJigsawPuzzle3D")
        n_frames = dataset_params.n_frames
        # SeparateJigsawPuzzle3D のスキーム設定をインスタンス化
        pretrain_scheme = SeparateJigsawPuzzle3D(
            img_size=dataset_params.img_size,
            n_frames=dataset_params.n_frames,
            mask_grid_ratio=pretrain_scheme_params.mask_grid_ratio,
            n_grid=pretrain_scheme_params.n_grid,
            mean=dataset_params.train_transforms.norm_mean,
            std=dataset_params.train_transforms.norm_std,
        )
        # 空間・時間分類のクラス数を取得
        num_classes_spatial = pretrain_scheme.n_classes_spatial
        num_classes_temporal = pretrain_scheme.n_classes_temporal

    elif pretrain_scheme_params.type.upper() == "JOINTJIGSAWPUZZLE3D":
        print("事前学習タイプ：JointJigsawPuzzle3D")
        n_frames = dataset_params.n_frames
        # JointJigsawPuzzle3D のスキーム設定をインスタンス化
        pretrain_scheme = JointJigsawPuzzle3D(
            img_size=dataset_params.img_size,
            n_frames=dataset_params.n_frames,
            mask_grid_ratio=pretrain_scheme_params.mask_grid_ratio,
            n_grid=pretrain_scheme_params.n_grid,
            mean=dataset_params.train_transforms.norm_mean,
            std=dataset_params.train_transforms.norm_std,
        )
        # 空間・時間分類のクラス数を取得
        num_classes_spatial = pretrain_scheme.n_classes
        num_classes_temporal = pretrain_scheme.n_classes

    else:
        # 未対応のスキームタイプの場合は例外をスロー
        raise Exception(f"未対応の事前学習スキームタイプ: {pretrain_scheme_params.type}")

    # 時間変換の設定を生成
    temporal_transforms = getTemporalTransforms(
        n_frames=n_frames,
        frame_interval=dataset_params.frame_interval,
    )

    # データセットタイプに基づいて事前学習用データセットを生成
    if dataset_params.datatype.upper() == "KINETICSK400":
        print("データタイプ：KineticsK400")
        # データセットパスの存在確認
        PreTrainDatasetK400.check_datapath(dataset_params.pretrain.pretrain_path)
        # PreTrainDatasetK400 インスタンスを生成
        pretrain_dataset = PreTrainDatasetK400(
            data_path=dataset_params.pretrain.pretrain_path,
            pretrain_scheme=pretrain_scheme,
            temporal_transforms=temporal_transforms,
            n_frames=n_frames,
            n_sample_per_class=dataset_params.pretrain.n_sample_per_class,
            movie_ext=dataset_params.movie_ext,
        )

    elif dataset_params.datatype.upper() == "UCF101":
        print("データタイプ：UCF101")
        # データセットパスの存在確認
        PreTrainDatasetUCF101.check_datapath(dataset_params.pretrain.pretrain_path)
        # PreTrainDatasetUCF101 インスタンスを生成
        pretrain_dataset = PreTrainDatasetUCF101(
            data_path=dataset_params.pretrain.pretrain_path,
            pretrain_scheme=pretrain_scheme,
            temporal_transforms=temporal_transforms,
            n_frames=n_frames,
            n_sample_per_class=dataset_params.pretrain.n_sample_per_class,
            movie_ext=dataset_params.movie_ext,
        )

    else:
        # 未対応のデータセットタイプの場合は例外をスロー
        raise Exception(f"未対応のデータセットタイプ: {dataset_params.datatype}")

    # 事前学習用データセットとクラス数を辞書形式で返す
    return {
        "pretrain_dataset": pretrain_dataset,
        "num_classes": (num_classes_spatial, num_classes_temporal),
    }

# メイン部分（動作確認用）
if __name__ == "__main__":
    # データセット初期化と生成のテスト
    initDatasetConfig("K400-Base")  # データセット設定を初期化
    datasets = getDataset()  # 学習用と検証用のデータセットを取得
    print(datasets)