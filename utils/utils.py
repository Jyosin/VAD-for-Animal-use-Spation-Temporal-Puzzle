import os
import yaml
import numpy as np 
import torch
import matplotlib.pyplot as plt
import urllib.request as req
import ast
import traceback
import datetime
import shutil
import inspect
from rich.console import Console
import subprocess

class TensorBoard():
    def __init__(self,logdir:str):
        self.logdir=logdir
        self.process=None

    def start(self,port:str="8080"):
        try: 
            self.process=subprocess.Popen(args=["tensorboard","--logdir",self.logdir,"--port",port],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            print("TensorBoardを下記で開始しました。")
            print(f"http://localhost:{port}/")
        except: 
            print("TensorBoardの起動に失敗しました。")
            traceback.format_exc()
        

    def end(self):
        try:
            self.process.kill()
            print("TensorBoard終了")
        except:
            traceback.format_exc()

def print(text: str = "", role: str = "text", offset_text: str = ""):
    """
    カスタムprint関数。リッチテキスト表示をサポート。
    
    Args:
        text (str): 表示するテキスト。
        role (str): 描画スタイル。"text" でそのまま表示、"rule" でラインを引く。
        offset_text (str): インデントに使用するテキスト。
    """
    console = Console()
    if role.upper() == "TEXT":
        n_stack = len(inspect.stack())
        offset = offset_text * (n_stack - 2)
        console.print(offset + text)
    elif role.upper() == "RULE":
        console.rule(text)

def get_pretrained_models(model_dest=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/pretrained')):
    """
    事前学習済みモデルをダウンロードして指定ディレクトリに保存する関数。

    Args:
        model_dest (str): モデルを保存するディレクトリパス。デフォルトは `models/pretrained`。

    Downloads:
        ViT, Swin Transformer, MViTなどの事前学習済みモデルを指定ディレクトリに保存する。
    """
    # ダウンロード対象モデルのURLを指定
    model_urls = {
        'vit_base_patch16_224-1k.pth': 'https://drive.google.com/file/d/1QjGpbR8K4Cf4TJaDc60liVhBvPtrc2v4/view?usp=sharing',
        'vit_base_patch16_224-21k.pth': 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/vit_base_patch16_224_miil_21k.pth',
        'swin_base_patch4_window7_224-1k.pth': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',
        'swin_base_patch4_window7_224-22k.pth': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
        'MViTv2_B_in1k.pyth': 'https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_B_in1k.pyth'
    }
    
    def progress_print(block_count, block_size, total_size):
        """
        ダウンロードの進行状況を表示する関数。

        Args:
            block_count (int): 現在までにダウンロードされたブロックの数。
            block_size (int): 各ブロックのサイズ（バイト単位）。
            total_size (int): ダウンロードする総データサイズ。
        """
        percentage = 100.0 * block_count * block_size / total_size
        if percentage > 100:
            percentage = 100
        max_bar = 100
        bar_num = int(percentage / (100 / max_bar))
        progress_element = '=' * bar_num
        if bar_num != max_bar:
            progress_element += '>'
        bar_fill = ' ' 
        bar = progress_element.ljust(max_bar, bar_fill)
        total_size_mb = total_size / (2**20)
        print(f'[{bar}] {percentage:.2f}% ( {total_size_mb:.2f}MB )\r', end='')

    # モデル保存用ディレクトリの存在確認と作成
    if not os.path.exists(model_dest): 
        os.makedirs(model_dest)
    
    # モデルのダウンロードを実行
    try:
        for filename, url in model_urls.items():
            if not os.path.exists(os.path.join(model_dest, filename)):
                print(f"Download pretrained model from {url}")
                req.urlretrieve(url, os.path.join(model_dest, filename), progress_print)
        print("Completed")
    except:
        print("Fail to download pretrained models...")
        print(traceback.format_exc())

class AttrDict():
    """
    辞書のキーを属性としてアクセスできるようにするクラス。

    Args:
        dictionary (dict): 初期化する辞書オブジェクト。
    """
    def __init__(self, dictionary: dict):
        # 辞書の内容を属性として設定
        for key, value in dictionary.items():
            if type(value) == dict:
                setattr(self, key, AttrDict(value))
            elif type(value) == str: 
                try:
                    setattr(self, key, ast.literal_eval(value))
                except:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)

def load_config(config_path) -> AttrDict:
    """
    設定ファイル（YAML）をロードし、辞書オブジェクトを作成する。データセットとモデルの設定も適用し、結果を保存する。

    Args:
        config_path (str): 設定ファイルのパス。

    Returns:
        tuple: 設定内容の辞書と結果を保存するディレクトリのパス。
    """
    assert os.path.exists(config_path), f"{config_path}がありません。"
    with open(config_path, 'r') as yml: 
        config = yaml.safe_load(yml)

    # 結果を保存するディレクトリを生成し、既存のディレクトリがある場合は削除してから作成
    result_dir = os.path.join(config["others"]["result_path"],config["others"]["label"])
    shutil.rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir, exist_ok=True)

    model_type = config["model"]["type"]
    dataset_type = config["dataset"]["type"]
    
    # データセットおよびモデルの設定を初期化
    from model.ModelFactory import initModelConfig
    from dataset.DatasetFactory import initDatasetConfig
    initModelConfig(model_type=model_type)
    initDatasetConfig(dataset_type=dataset_type)
    
    # 読み込んだ設定内容を結果ディレクトリに保存
    with open(os.path.join(result_dir, "config.yml"), mode="w", encoding="utf-8") as f:
        yaml.safe_dump(
            data=config,
            stream=f,
            indent=4,
            allow_unicode=True,
            sort_keys=False
        )

    return AttrDict(config), result_dir

def load_yml(config_path='./config.yml') -> AttrDict:
    """
    YAML設定ファイルを読み込み、属性アクセスが可能な辞書オブジェクトを返す。

    Args:
        config_path (str): 読み込む設定ファイルのパス。

    Returns:
        AttrDict: 読み込んだ設定内容を属性としてアクセスできる辞書オブジェクト。
    """
    assert os.path.exists(config_path), f"{config_path}がありません。"
    with open(config_path, 'r') as yml: 
        config = yaml.safe_load(yml)
    return AttrDict(config)

def imsave(img_array, save_path):
    """
    画像データを保存するための関数。テンソルをNumpy配列に変換し、画像ファイルとして保存。

    Args:
        img_array (torch.Tensor or np.array): 保存する画像データ。
        save_path (str): 保存先のパス。
    """
    # テンソルをNumpy配列に変換（必要に応じて軸の並び替えとクリッピングを行う）
    if type(img_array) == torch.Tensor:
        img_array = img_array.cpu().detach().numpy().transpose(1, 2, 0)
    if np.max(img_array) == 1:
        img_array = np.clip(img_array * 255, a_min=0, a_max=255).astype(np.uint8)
    
    # 画像を表示し、指定のパスに保存
    plt.imshow(img_array)
    plt.savefig(save_path)
    plt.close()