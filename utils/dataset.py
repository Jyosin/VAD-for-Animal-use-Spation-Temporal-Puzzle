import decord
import torch
import glob, json, os, tqdm
from pathlib import Path
from utils.utils import imsave, print
from dataset.pretrain_scheme.pretrain_schemes.CubicPuzzle3D import CubicPuzzle3D
from dataset.pretrain_scheme.pretrain_schemes.SeparateJigsawPuzzle3D import SeparateJigsawPuzzle3D
from dataset.pretrain_scheme.pretrain_schemes.JointJigsawPuzzle3D import JointJigsawPuzzle3D
import torch.utils.data

def load_annotation_data(data_file_path):
    """
    アノテーションデータ（JSON形式）を読み込む関数。

    Args:
        data_file_path (str): JSONファイルのパス。

    Returns:
        dict: JSONデータを辞書形式で返す。
    """
    with open(data_file_path, 'r') as data_file:  # 指定されたJSONファイルを開く。
        return json.load(data_file)  # ファイル内容をJSON形式としてロード。

def load_data(data_path, n_samples_per_cls, movie_ext='.mp4'):
    """
    データセットからデータを読み込み、クラスラベルと動画のパスを整理する。

    Args:
        data_path (str): データセットのパス。
        n_samples_per_cls (int, float): 各クラスからサンプリングするデータ数の割合。
        movie_ext (str): 動画ファイルの拡張子（デフォルトは '.mp4'）。

    Returns:
        tuple: (data_dicts, labels) 形式でデータとラベル情報を返す。
    """
    data_pathes = sorted(glob.glob(data_path + "/*/*" + movie_ext))

    class_to_idx = load_annotation_data(data_path + "/classmap.json")  # クラスラベルマップの読み込み
    data_dicts = []
    labels = {}

    for path in data_pathes:
        class_name = Path(path).parent.name
        # 新しいクラス名の場合、初期設定
        if not class_name in list(labels.keys()): 
            labels[class_name] = {"count": 1, "n_sample_per_class": None}
            if type(n_samples_per_cls) == int:
                labels[class_name]["n_sample_per_class"] = n_samples_per_cls
            elif type(n_samples_per_cls) == float:
                # クラスごとに指定割合でサンプル数を設定
                labels[class_name]["n_sample_per_class"] = int(len(glob.glob(data_path + f"/{class_name}/*" + movie_ext)) * n_samples_per_cls)
        # サンプル数の上限に達した場合は次の動画に進む
        elif labels[class_name]["n_sample_per_class"]:
            if labels[class_name]["n_sample_per_class"] <= labels[class_name]["count"]:
                continue
            else:
                labels[class_name]["count"] += 1

        class_idx = int(class_to_idx[class_name])  # クラスラベルのインデックスを取得
        data_dicts.append({'video': path, 'label': class_idx})

    return data_dicts, labels

class DecordInit:
    """Decordを使用して動画リーダーを初期化するためのクラス。"""

    def __init__(self, num_threads=1, **kwargs):
        """
        動画リーダーの初期化設定。

        Args:
            num_threads (int): 使用するスレッド数。
            **kwargs: その他の追加設定。
        """
        self.num_threads = num_threads
        self.kwargs = kwargs

    def __call__(self, filename):
        """
        Decordを使用して動画リーダーを生成する。

        Args:
            filename (str): 動画ファイルのパス。

        Returns:
            decord.VideoReader: 初期化された動画リーダー。
        """
        return decord.VideoReader(filename, ctx=decord.cpu(0), num_threads=self.num_threads)

    def __repr__(self):
        """オブジェクトの文字列表現を返す。"""
        return f"{self.__class__.__name__}(num_threads={self.num_threads})"

def data_sampling(dataset: torch.utils.data.Dataset, save_path: str, n_samples: int = 5):
    """
    データセットから指定数のサンプルを抽出して保存する関数。
    オリジナルフレームと分類タスク用の入力データを保存。

    Args:
        dataset (torch.utils.data.Dataset): 対象のデータセット。
        save_path (str): 保存先のディレクトリ。
        n_samples (int): 抽出するサンプル数。
    """
    dataset.spatial_transform.transforms = dataset.spatial_transform.transforms[:-1]  # Normalizeを除去。

    for i in range(n_samples):
        os.makedirs(f"{save_path}/{i+1:02d}/original", exist_ok=True)  # オリジナルフレーム保存用フォルダを作成。
        os.makedirs(f"{save_path}/{i+1:02d}/cls_input", exist_ok=True)  # 入力データ保存用フォルダを作成。

        orig_video = dataset.get_original_video(i)  # 元動画データを取得。
        for j, v in enumerate(orig_video[:64]):  # 最初の64フレームを保存。
            imsave(v, f"{save_path}/{i+1:02d}/original/{j:03d}.png")
        print(f"オリジナルフレームデータサンプル出力 -> {save_path}/{i+1:02d}/original")

        inputs, labels = dataset[i]  # サンプルの入力とラベルを取得。

        for j, v in enumerate(inputs):  # 入力データを保存。
            imsave(v, f"{save_path}/{i+1:02d}/cls_input/{j:03d}.png")
        print(f"分類タスク学習入力データサンプル出力 -> {save_path}/{i+1:02d}/cls_input")
        print()

def pretrain_data_sampling(dataset: torch.utils.data.Dataset, save_path: str, n_samples: int = 5):
    """
    事前学習データセットから指定数のサンプルを抽出して保存する関数。
    オリジナルフレーム、空間シャッフル、時間シャッフルのサンプルを保存。

    Args:
        dataset (torch.utils.data.Dataset): 対象のデータセット。
        save_path (str): 保存先のディレクトリ。
        n_samples (int): 抽出するサンプル数。
    """
    dataset.pretrain_scheme.transforms.transforms = dataset.pretrain_scheme.transforms.transforms[:-1]  # Normalizeを除去。

    if isinstance(dataset.pretrain_scheme, CubicPuzzle3D):
        pretrain_data_sampling_cubicpuzzle3d(dataset=dataset, save_path=save_path, n_samples=n_samples)
    elif isinstance(dataset.pretrain_scheme, SeparateJigsawPuzzle3D):
        pretrain_data_sampling_separatejigsawpuzzle3d(dataset=dataset, save_path=save_path, n_samples=n_samples)
    elif isinstance(dataset.pretrain_scheme, JointJigsawPuzzle3D):
        pretrain_data_sampling_jointjigsawpuzzle3d(dataset=dataset, save_path=save_path, n_samples=n_samples)

def pretrain_data_sampling_cubicpuzzle3d(dataset: torch.utils.data.Dataset, save_path: str, n_samples: int = 5) -> None:
    """
    CubicPuzzle3Dの事前学習用データセットからサンプルを抽出して保存する関数。

    Args:
        dataset (torch.utils.data.Dataset): 対象のデータセット。
        save_path (str): サンプルを保存するディレクトリ。
        n_samples (int): 保存するサンプル数。
    """
    for i in range(n_samples):
        # 保存先ディレクトリを作成
        os.makedirs(f"{save_path}/{i+1:02d}/original", exist_ok=True)
        os.makedirs(f"{save_path}/{i+1:02d}/pretrain_input_spatial", exist_ok=True)
        os.makedirs(f"{save_path}/{i+1:02d}/pretrain_input_temporal", exist_ok=True)

        # 元の動画データを取得し、保存
        orig_video = dataset.get_original_video(i)
        for j, v in enumerate(orig_video[:64]):  # 最初の64フレームを保存
            imsave(v, f"{save_path}/{i+1:02d}/original/{j:03d}.png")
        print(f"[事前学習] オリジナルフレームデータ出力 -> {save_path}/{i+1:02d}/original")

        # 空間シャッフルデータを保存
        spatial_inputs, temporal_inputs = dataset[i]
        tensor, labels = spatial_inputs
        for j, v in enumerate(tensor):  # 各空間データを保存
            for k, _v in enumerate(v):
                imsave(_v, f"{save_path}/{i+1:02d}/pretrain_input_spatial/{j:03d}-{k:03d}.png")
        print(f"[事前学習] 空間シャッフルサンプルデータ出力 -> {save_path}/{i+1:02d}/pretrain_input_spatial")

        # 時間シャッフルデータを保存
        tensor, labels = temporal_inputs
        for j, v in enumerate(tensor):  # 各時間データを保存
            for k, _v in enumerate(v):
                imsave(_v, f"{save_path}/{i+1:02d}/pretrain_input_temporal/{j:03d}-{k:03d}.png")
        print(f"[事前学習] 時間シャッフルサンプルデータ出力 -> {save_path}/{i+1:02d}/pretrain_input_temporal")
        print()

def pretrain_data_sampling_separatejigsawpuzzle3d(dataset: torch.utils.data.Dataset, save_path: str, n_samples: int = 5) -> None:
    """
    SeparateJigsawPuzzle3Dの事前学習用データセットからサンプルを抽出して保存する関数。

    Args:
        dataset (torch.utils.data.Dataset): 対象のデータセット。
        save_path (str): サンプルを保存するディレクトリ。
        n_samples (int): 保存するサンプル数。
    """
    for i in range(n_samples):
        # 保存先ディレクトリを作成
        os.makedirs(f"{save_path}/{i+1:02d}/original", exist_ok=True)
        os.makedirs(f"{save_path}/{i+1:02d}/pretrain_input_spatial", exist_ok=True)
        os.makedirs(f"{save_path}/{i+1:02d}/pretrain_input_temporal", exist_ok=True)

        # 元の動画データを取得し、保存
        orig_video = dataset.get_original_video(i)
        for j, v in enumerate(orig_video[:64]):  # 最初の64フレームを保存
            imsave(v, f"{save_path}/{i+1:02d}/original/{j:03d}.png")
        print(f"[事前学習] オリジナルフレームデータ出力 -> {save_path}/{i+1:02d}/original")

        # 空間マスクデータを保存
        spatial_inputs, temporal_inputs = dataset[i]
        tensor, labels = spatial_inputs
        for j, v in enumerate(tensor):  # 各空間データを保存
            imsave(v, f"{save_path}/{i+1:02d}/pretrain_input_spatial/{j:03d}.png")
        print(f"[事前学習] 空間マスクサンプルデータ出力 -> {save_path}/{i+1:02d}/pretrain_input_spatial")

        # 時間マスクデータを保存
        tensor, labels = temporal_inputs
        for j, v in enumerate(tensor):  # 各時間データを保存
            imsave(v, f"{save_path}/{i+1:02d}/pretrain_input_temporal/{j:03d}.png")
        print(f"[事前学習] 時間マスクサンプルデータ出力 -> {save_path}/{i+1:02d}/pretrain_input_temporal")
        print()

def pretrain_data_sampling_jointjigsawpuzzle3d(dataset: torch.utils.data.Dataset, save_path: str, n_samples: int = 5) -> None:
    """
    JointJigsawPuzzle3Dの事前学習用データセットからサンプルを抽出して保存する関数。

    Args:
        dataset (torch.utils.data.Dataset): 対象のデータセット。
        save_path (str): サンプルを保存するディレクトリ。
        n_samples (int): 保存するサンプル数。
    """
    for i in range(n_samples):
        # 保存先ディレクトリを作成
        os.makedirs(f"{save_path}/{i+1:02d}/original", exist_ok=True)
        os.makedirs(f"{save_path}/{i+1:02d}/pretrain_input", exist_ok=True)

        # 元の動画データを取得し、保存
        orig_video = dataset.get_original_video(i)
        for j, v in enumerate(orig_video[:64]):  # 最初の64フレームを保存
            imsave(v, f"{save_path}/{i+1:02d}/original/{j:03d}.png")
        print(f"[事前学習] オリジナルフレームデータ出力 -> {save_path}/{i+1:02d}/original")

        # 時空間マスクデータを保存
        spatial_inputs, temporal_inputs = dataset[i]
        tensor, labels = spatial_inputs
        for j, v in enumerate(tensor):  # 各データを保存
            imsave(v, f"{save_path}/{i+1:02d}/pretrain_input/{j:03d}.png")
        print(f"[事前学習] 時空間マスクサンプルデータ出力 -> {save_path}/{i+1:02d}/pretrain_input")
        print()