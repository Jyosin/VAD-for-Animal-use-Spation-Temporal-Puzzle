import random
import numpy as np
import torch
from utils.dataset import DecordInit, load_data  # 動画デコードとデータロード用
import requests  # HTTPリクエスト用
import os  # ファイル操作用
from rich.progress import track  # プログレスバー表示用

class DatasetUCF101(torch.utils.data.Dataset):
    """
    UCF101 データセットを読み込むためのクラス。

    動画データを空間変換および時間方向のサンプリングで処理し、
    モデル学習用に前処理されたデータを提供します。

    Args:
        data_path (str): データセットのパス。
        spatial_transform (callable): 空間変換を行う関数。
        temporal_transform (callable): 時間方向のサンプリングを行う関数。
        n_frames (int): 読み込むフレーム数。
        n_sample_per_class (float): 各クラスからサンプリングするデータ数の割合。
        movie_ext (str): 動画ファイルの拡張子（デフォルトは ".mp4"）。
    """
    
    def __init__(self, data_path, spatial_transform, temporal_transform, n_frames, n_sample_per_class=1.0, movie_ext=".mp4"):
        # 動画データの読み込み
        self.data, _ = load_data(data_path=data_path, n_samples_per_cls=n_sample_per_class, movie_ext=movie_ext)
        self.spatial_transform = spatial_transform  # 空間変換用関数
        self.temporal_transform = temporal_transform  # 時間変換用関数
        self.target_video_len = n_frames  # 必要なフレーム数
        self.v_decoder = DecordInit()  # 動画デコーダの初期化
    
    def __getitem__(self, index):
        """
        指定されたインデックスのデータを取得し、前処理して返す。

        Args:
            index (int): データセットのインデックス。
        
        Returns:
            tuple: (変換されたビデオデータ (torch.Tensor), ラベル (int))
        """
        while True:
            try:
                path = self.data[index]['video']  # 動画ファイルのパス
                v_reader = self.v_decoder(path)  # 動画デコード
                total_frames = len(v_reader)  # 総フレーム数を取得

                # 時間方向のフレームをサンプリング
                start_frame_ind, end_frame_ind = self.temporal_transform(total_frames)
                assert end_frame_ind - start_frame_ind >= self.target_video_len, "フレーム数が不足しています"
                frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
                video = v_reader.get_batch(frame_indice).asnumpy()  # 指定フレームを取得
                del v_reader  # メモリ解放
                break
            except Exception:
                # 失敗時はランダムに新しいインデックスを取得
                index = random.randint(0, len(self.data) - 1)
        
        # 動画データを (T, C, H, W) に変換し、空間変換を適用
        with torch.no_grad():
            video = torch.from_numpy(video).permute(0, 3, 1, 2)  # 次元を変更
            video = self.spatial_transform(video)  # 空間変換を適用
            label = self.data[index]['label']  # ラベルを取得
            return video, label

    def __len__(self):
        """
        データセットのサイズを返す。

        Returns:
            int: データセットのサイズ。
        """
        return len(self.data)
    
    def get_original_video(self, index):
        """
        元の動画を取得します（前処理なし）。

        Args:
            index (int): データセットのインデックス。

        Returns:
            torch.Tensor: 元のフレームデータ。
        """
        while True:
            try:
                path = self.data[index]['video']  # 動画ファイルのパス
                v_reader = self.v_decoder(path)  # 動画デコード
                total_frames = len(v_reader)  # 総フレーム数を取得
                video = v_reader.get_batch(np.arange(total_frames)).asnumpy()  # 全フレームを取得
                del v_reader  # メモリ解放
                break
            except Exception:
                # 失敗時はランダムに新しいインデックスを取得
                index = random.randint(0, len(self.data) - 1)

        return torch.from_numpy(video).permute(0, 3, 1, 2)  # (T, C, H, W) に変換

    @staticmethod
    def check_datapath(datapath, mode="train"):
        """
        データパスが存在するか確認し、存在しない場合はダウンロードを行う。

        Args:
            datapath (str): データセットの保存先パス。
            mode (str): "train" または "valid"（デフォルトは "train"）。
        """
        if not os.path.exists(datapath):
            print(f"{datapath} が存在しません。データをダウンロードします。")
            loader = UCF101Downloader()
            loader.download(datapath)


class PreTrainDatasetUCF101(torch.utils.data.Dataset):
    """
    UCF101 データセットを事前学習用に読み込むクラス。

    動画データに事前学習用の変換（空間および時間方向のパズル）を適用します。

    Args:
        data_path (str): データセットのパス。
        pretrain_scheme (callable): 空間および時間方向のパズル変換を行う関数。
        temporal_transforms (callable): 時間方向のサンプリングを行う関数。
        n_frames (int): 読み込むフレーム数。
        n_sample_per_class (float): 各クラスからサンプリングするデータ数の割合。
        movie_ext (str): 動画ファイルの拡張子（デフォルトは ".mp4"）。
    """
    def __init__(self, data_path, pretrain_scheme, temporal_transforms, n_frames, n_sample_per_class=1.0, movie_ext=".mp4"):
        # 動画データの読み込み
        self.data, _ = load_data(data_path=data_path, n_samples_per_cls=n_sample_per_class, movie_ext=movie_ext)
        self.v_decoder = DecordInit()  # 動画デコーダ
        self.temporal_transform = temporal_transforms  # 時間方向のサンプリング変換
        self.pretrain_scheme = pretrain_scheme  # 事前学習用の変換
        self.target_video_len = n_frames  # 必要なフレーム数

    def __getitem__(self, index):
        """
        指定インデックスのデータに事前学習用変換を適用し取得する。

        Args:
            index (int): データセットのインデックス。

        Returns:
            tuple: 空間および時間方向のパズル変換後のデータ。
        """
        while True:
            try:
                path = self.data[index]['video']  # 動画ファイルのパス
                v_reader = self.v_decoder(path)  # 動画デコード
                total_frames = len(v_reader)  # フレーム数を取得

                start_frame_ind, end_frame_ind = self.temporal_transform(total_frames)
                assert end_frame_ind - start_frame_ind >= self.target_video_len
                frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
                video = v_reader.get_batch(frame_indice).asnumpy()  # 指定フレームを取得
                del v_reader
                break
            except Exception:
                # ランダムに別のインデックスを取得
                index = random.randint(0, len(self.data) - 1)

        # 動画データに空間および時間方向の変換を適用
        with torch.no_grad():
            video = torch.from_numpy(video).permute(0, 3, 1, 2)
            spatial_puzzle, temporal_puzzle = self.pretrain_scheme(video)
            return spatial_puzzle, temporal_puzzle

    def __len__(self):
        """
        データセットのサイズを返す。

        Returns:
            int: データセットのサイズ。
        """
        return len(self.data)

    def get_original_video(self, index):
        """
        元の動画データを取得します（前処理なし）。

        Args:
            index (int): データセットのインデックス。

        Returns:
            torch.Tensor: 元のフレームデータ。
        """
        while True:
            try:
                path = self.data[index]['video']  # 動画ファイルのパス
                v_reader = self.v_decoder(path)  # 動画デコード
                total_frames = len(v_reader)  # 総フレーム数を取得
                video = v_reader.get_batch(np.arange(total_frames)).asnumpy()  # 全フレームを取得
                del v_reader
                break
            except Exception:
                index = random.randint(0, len(self.data) - 1)

        return torch.from_numpy(video).permute(0, 3, 1, 2)  # (T, C, H, W) に変換

    @staticmethod
    def check_datapath(datapath):
        """
        データパスが存在するか確認。

        Args:
            datapath (str): データセットの保存先パス。

        Raises:
            Exception: パスが存在しない場合にエラーを発生させる。
        """
        if not os.path.exists(datapath):
            raise Exception(f"{datapath} が存在しません。")


class UCF101Downloader:
    """
    UCF101 データセットの動画をダウンロードするクラス。

    Args:
        url (str): データセットのダウンロードURL（デフォルトは公式サイトのUCF101）。
    """
    url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    
    def __init__(self):
        self.progress = None
        self.task = None

    def download(self, out_dir, chunk_size=1024):
        """
        データセットの動画をダウンロードします。

        Args:
            out_dir (str): 動画の保存先ディレクトリ。
            chunk_size (int): ダウンロード時のチャンクサイズ（デフォルトは 1024）。
        """
        os.makedirs(out_dir, exist_ok=True)  # 保存先ディレクトリの作成
        file_name = os.path.basename(self.url)  # 保存するファイル名を取得
        file_size = int(requests.head(self.url, verify=False).headers["content-length"])  # ファイルサイズを取得
        res = requests.get(self.url, stream=True, verify=False)  # データをストリームで取得

        if res.status_code == 200:
            iterator = res.iter_content(chunk_size=chunk_size)
            with open(os.path.join(out_dir, file_name), 'wb') as file:
                for chunk in track(iterator, description=f"Downloading {file_name}", total=(file_size // chunk_size) + 1):
                    file.write(chunk)