import random
import numpy as np
import torch
from utils.dataset import DecordInit, load_data  # 動画デコードとデータロード用
from pytubefix import YouTube  # YouTube動画のダウンロードライブラリ
from pytubefix.exceptions import BotDetection  # Bot検出時の例外
import ffmpeg, json, tqdm, traceback, os  # 必要なライブラリ
from concurrent.futures import ThreadPoolExecutor, as_completed  # 並列処理
import urllib.request as req  # HTTPリクエスト用
from utils.utils import print  # カスタムprint関数

class DatasetK400(torch.utils.data.Dataset):
    """
    Kinetics 400 データセットを読み込むためのクラス。
    
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
        self.spatial_transform = spatial_transform  # 空間変換
        self.temporal_transform = temporal_transform  # 時間方向の変換
        self.target_video_len = n_frames  # 必要なフレーム数
        self.v_decoder = DecordInit()  # 動画デコーダの初期化

    def __getitem__(self, index):
        """
        指定されたインデックスのデータを取得し、前処理して返す。

        Args:
            index (int): データセットのインデックス。

        Returns:
            tuple: (動画データ (torch.Tensor), ラベル (int))
        """
        while True:
            try:
                path = self.data[index]['video']  # 動画のパス
                v_reader = self.v_decoder(path)  # 動画デコーダ
                total_frames = len(v_reader)  # 総フレーム数の取得

                # 時間方向のサンプリング
                start_frame_ind, end_frame_ind = self.temporal_transform(total_frames)
                assert end_frame_ind - start_frame_ind >= self.target_video_len, "フレーム数が不足しています"
                frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
                video = v_reader.get_batch(frame_indice).asnumpy()  # 指定フレームを取得
                del v_reader
                break
            except Exception:
                # 失敗した場合は別のインデックスをランダムに取得
                index = random.randint(0, len(self.data) - 1)

        # 動画データを (T, C, H, W) に変換し、空間変換を適用
        with torch.no_grad():
            video = torch.from_numpy(video).permute(0, 3, 1, 2)  # 次元を変換
            video = self.spatial_transform(video)  # 空間変換
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
        元の動画データを取得します（前処理なし）。

        Args:
            index (int): データセットのインデックス。

        Returns:
            torch.Tensor: 元のフレームデータ。
        """
        while True:
            try:
                path = self.data[index]['video']  # 動画のパス
                v_reader = self.v_decoder(path)  # デコード
                total_frames = len(v_reader)  # フレーム数を取得
                video = v_reader.get_batch(np.arange(total_frames)).asnumpy()  # 全フレームを取得
                del v_reader
                break
            except Exception:
                # 失敗時は新しいインデックスを選択
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
            if mode.upper() == "TRAIN":
                loader = KineticsDownloader(num_workers=10, mode="train")
                loader.download(datapath)
            elif mode.upper() == "VALID":
                loader = KineticsDownloader(num_workers=10, mode="valid")
                loader.download(datapath)


class PreTrainDatasetK400(torch.utils.data.Dataset):
    """
    Kinetics 400 データセットを事前学習用に読み込むクラス。

    動画データに事前学習用の変換（パズル）を適用します。

    Args:
        data_path (str): データセットのパス。
        pretrain_scheme (callable): 空間および時間方向のパズル変換を行う関数。
        temporal_transforms (callable): 時間方向のサンプリングを行う関数。
        n_frames (int): 読み込むフレーム数。
        n_sample_per_class (float): 各クラスからサンプリングするデータ数の割合。
        movie_ext (str): 動画ファイルの拡張子（デフォルトは ".mp4"）。
    """
    def __init__(self, data_path, pretrain_scheme, temporal_transforms, n_frames, n_sample_per_class=1.0, movie_ext=".mp4"):
        self.data, _ = load_data(data_path=data_path, n_samples_per_cls=n_sample_per_class, movie_ext=movie_ext)
        self.v_decoder = DecordInit()  # 動画デコーダ
        self.temporal_transform = temporal_transforms  # 時間方向のサンプリング変換
        self.pretrain_scheme = pretrain_scheme  # 事前学習用パズル変換
        self.target_video_len = n_frames  # 必要なフレーム数

    def __getitem__(self, index):
        """
        指定インデックスのデータに事前学習用変換を適用し取得する。

        Args:
            index (int): データセットのインデックス。

        Returns:
            tuple: 空間および時間方向のパズル変換後データ。
        """
        while True:
            try:
                path = self.data[index]['video']
                v_reader = self.v_decoder(path)
                total_frames = len(v_reader)

                start_frame_ind, end_frame_ind = self.temporal_transform(total_frames)
                assert end_frame_ind - start_frame_ind >= self.target_video_len
                frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int)
                video = v_reader.get_batch(frame_indice).asnumpy()
                del v_reader
                break
            except Exception:
                index = random.randint(0, len(self.data) - 1)

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
        元の動画を取得する（前処理なし）。

        Args:
            index (int): データセットのインデックス。

        Returns:
            torch.Tensor: 元のフレームデータ。
        """
        while True:
            try:
                path = self.data[index]['video']
                v_reader = self.v_decoder(path)
                total_frames = len(v_reader)
                video = v_reader.get_batch(np.arange(total_frames)).asnumpy()
                del v_reader
                break
            except Exception:
                index = random.randint(0, len(self.data) - 1)

        return torch.from_numpy(video).permute(0, 3, 1, 2)

    @staticmethod
    def check_datapath(datapath):
        """
        データパスの存在確認。

        Args:
            datapath (str): データのパス。

        Raises:
            Exception: パスが存在しない場合にエラーを発生させる。
        """
        if not os.path.exists(datapath):
            raise Exception(f"{datapath} が存在しません。")


class KineticsDownloader:
    """
    Kinetics 400 データセットの動画を YouTube からダウンロードするクラス。

    Args:
        mode (str): データのモード ("train"または"valid")。
        num_workers (int): 並行してダウンロードするワーカ数。
    """
    base_url = "http://youtube.com/watch?v="
    url_dict = {
        'train': "https://s3.amazonaws.com/kinetics/400/annotations/train.csv",
        'valid': "https://s3.amazonaws.com/kinetics/400/annotations/val.csv"
    }

    def __init__(self, mode="train", num_workers=3):
        if not os.path.exists(f"./kinetics-400_{mode}.csv"):
            req.urlretrieve(self.url_dict[mode], f"./kinetics-400_{mode}.csv")
        
        self.csv_path = f"./kinetics-400_{mode}.csv"
        self.mode = mode
        self.threadpool = ThreadPoolExecutor(max_workers=num_workers)

    def download(self, out_dir):
        """
        データセットの動画をダウンロードする。

        Args:
            out_dir (str): 動画の出力先ディレクトリ。
        """
        with open(self.csv_path, mode="r") as f:
            lines = f.readlines()

        classmap = {}
        futures = []
        for line in lines[1:]:
            line = line.replace("\"", "")
            id = line.split(",")[1]
            label = line.split(",")[0]
            start_time = int(line.split(",")[2])
            end_time = int(line.split(",")[3])
            if label not in classmap.keys():
                classmap[label] = len(classmap)
            futures.append(self.threadpool.submit(self._download, id, label, start_time, end_time, out_dir))

        success = ""
        failed = ""
        cnt_failed = 0
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Downloading videos"):
            try:
                result = future.result()
                success += result
            except Exception as e:
                msg = str(e) + "\n" + traceback.format_exc() + "\n"
                failed += msg
                cnt_failed += 1

        with open(f"download-{self.mode}.log", mode="w") as f:
            f.write(f"{cnt_failed}/{len(lines[1:])} videos failed to download\n")
            f.write("[DOWNLOAD FAILED]\n")
            f.write(failed)
            f.write("[DOWNLOAD SUCCESSFULLY]\n")
            f.write(success)

        with open(f'{out_dir}/classmap.json', 'w') as f:
            json.dump(classmap, f, indent=2)

        self.threadpool.shutdown(wait=True)

    def _download(self, id, label, start_time, end_time, out_dir):
        """
        YouTube 動画をダウンロードし、指定された時間でトリミングする。

        Args:
            id (str): YouTube動画のID。
            label (str): 動画のクラスラベル。
            start_time (int): トリミング開始時間。
            end_time (int): トリミング終了時間。
            out_dir (str): 動画の出力ディレクトリ。

        Returns:
            str: ダウンロード結果のメッセージ。
        """
        out_path = os.path.join(out_dir, label)
        os.makedirs(out_path, exist_ok=True)
        try:
            if os.path.exists(out_path + f"/{id}.mp4"):
                return f"youtube_id ({id}) has already downloaded -> {out_path}/{id}.mp4\n"
            YouTube(self.base_url + id).streams.filter(progressive=True, file_extension='mp4').get_highest_resolution().download(out_path, f"_{id}.mp4")
            inp = ffmpeg.input(out_path + f"/_{id}.mp4")
            video = inp.trim(start=start_time, end=end_time).setpts('PTS-STARTPTS')
            audio = inp.filter('atrim', start=start_time, end=end_time).filter('asetpts', 'PTS-STARTPTS')
            ffmpeg.output(video, audio, out_path + f"/{id}.mp4").overwrite_output().run(quiet=True)
            os.remove(out_path + f"/_{id}.mp4")
            return f"youtube_id ({id}) has successfully downloaded -> {out_path}/{id}.mp4\n"
        except BotDetection as e:
            # Bot detectionが発生した場合、再試行
            return self._download(id, label, start_time, end_time, out_dir)
        except:
            raise Exception(f"Error occurred when downloading from {self.base_url + id}")