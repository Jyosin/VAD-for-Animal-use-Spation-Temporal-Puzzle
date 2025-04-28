import os, sys
import random
import warnings
import argparse
import multiprocessing
import gc

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.utils.data
from utils.utils import load_config, TensorBoard

from model.ModelFactory import getModel, getClsHead, getPretrainClsHead
from dataset.DatasetFactory import getDataset, getPreTrainDataset
from utils.datamodule import DataModule, PreTrainDataModule
from utils.dataset import data_sampling, pretrain_data_sampling
from trainers.trainer import TrainModule
from trainers.pretrainer import PreTrainModule
from utils.analysis import Analyzer
from utils.model_summary import RichModelSummary
from rich.console import Console

# 演算精度の設定（速度とメモリ効率のために float32 の精度を "medium" に設定）
torch.set_float32_matmul_precision("medium")


def pretrain(
        model: torch.nn.Module,
        spatial_cls_head: torch.nn.Module,
        temporal_cls_head: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 8,
        num_workers: int = 4,
        epoch: int = 10,
        optimizer: str = "adamw",
        lr: float = 0.0005,
        weight_decay: float = 0.05,
        label: str = "EXPERIMENT",
        log_dir: str = "./logs",
        result_dir: str = "./results"
) -> torch.nn.Module:
    """
    事前学習を行う関数。モデルと分類ヘッドを用いて、指定されたデータセットでの学習を実行する。

    Args:
        model (torch.nn.Module): 事前学習に使用するモデル。
        spatial_cls_head (torch.nn.Module): 空間的な分類ヘッド。
        temporal_cls_head (torch.nn.Module): 時間的な分類ヘッド。
        dataset (torch.utils.data.Dataset): 事前学習用データセット。
        batch_size (int): バッチサイズ。
        num_workers (int): データローダのワーカ数。
        epoch (int): エポック数。
        optimizer (str): 最適化手法（例："adamw"）。
        lr (float): 学習率。
        weight_decay (float): Weight decayの値。
        label (str): 実験のラベル（ログの識別に使用）。
        log_dir (str): ログの出力ディレクトリ。
        result_dir (str): 結果の保存ディレクトリ。

    Returns:
        torch.nn.Module: 事前学習後のモデル。
    """
    # TensorBoardロガーの設定（学習ログを保存するディレクトリを指定）
    logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, f"pretrain-{label}"))

    # データモジュールの初期化（データローダなどを管理するクラス）
    pretrain_data_module = PreTrainDataModule(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 事前学習モジュールの初期化（学習ループや損失関数、最適化アルゴリズムを管理）
    pretrain_module = PreTrainModule(
        model=model,
        spatial_cls_head=spatial_cls_head,
        temporal_cls_head=temporal_cls_head,
        optimizer=optimizer,
        lr=lr * batch_size / 256,  # バッチサイズに応じた学習率の調整
        weight_decay=weight_decay,
        result_dir=result_dir
    )

    # PyTorch Lightningのトレーナー設定
    pretrainer = pl.Trainer(
        accelerator="cuda",  # GPUでの学習を指定
        precision=16,  # 半精度演算の設定（メモリ節約と学習速度向上のため）
        max_epochs=epoch,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_model_summary=False,
        callbacks=RichModelSummary(model_sum_file=os.path.join(result_dir, "model_summary_pretrain.txt")),
        logger=logger
    )

    # 学習の実行
    pretrainer.fit(pretrain_module, pretrain_data_module)

    # 不要になったオブジェクトを削除してメモリを解放
    del pretrain_data_module, pretrain_module, pretrainer


def train(
        model: torch.nn.Module,
        cls_head: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        batch_size: int = 8,
        num_workers: int = 4,
        epoch: int = 10,
        optimizer: str = "adamw",
        lr: float = 0.0005,
        weight_decay: float = 0.05,
        label: str = "EXPERIMENT",
        log_dir: str = "./logs",
        result_dir: str = "./results"
        dataset_name : str = "unknown"
):
    """
    モデルの学習を行う関数。指定されたデータセットと分類ヘッドを用いて、モデルの学習を実行する。

    Args:
        model (torch.nn.Module): 学習するモデル。
        cls_head (torch.nn.Module): 分類ヘッド。
        train_dataset (torch.utils.data.Dataset): 学習用データセット。
        valid_dataset (torch.utils.data.Dataset): 検証用データセット。
        batch_size (int): バッチサイズ。
        num_workers (int): データローダのワーカ数。
        epoch (int): エポック数。
        optimizer (str): 最適化手法（例："adamw"）。
        lr (float): 学習率。
        weight_decay (float): Weight decayの値。
        label (str): 実験のラベル（ログの識別に使用）。
        log_dir (str): ログの出力ディレクトリ。
        result_dir (str): 結果の保存ディレクトリ。
    """

    # TensorBoardロガーの設定（学習ログを保存するディレクトリを指定）
    logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, f"{label}"))

    # データモジュールの初期化（学習データセットと検証データセットのデータローダ設定）
    data_module = DataModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 学習モジュールの初期化（学習ループや損失関数、最適化アルゴリズムを管理）
    train_module = TrainModule(
        model=model,
        cls_head=cls_head,
        optimizer=optimizer,
        lr=lr * batch_size / 256,  # バッチサイズに応じた学習率の調整
        weight_decay=weight_decay,
        result_dir=result_dir
    )

    # PyTorch Lightningのトレーナー設定
    trainer = pl.Trainer(
        accelerator="cuda",  # GPUでの学習を指定
        precision=16,  # 半精度演算の設定（メモリ節約と学習速度向上のため）
        max_epochs=epoch,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_model_summary=False,
        callbacks=RichModelSummary(model_sum_file=os.path.join(result_dir, "model_summary.txt")),
        logger=logger
    )

    # 学習の実行
    trainer.fit(train_module, data_module)

    # 不要になったオブジェクトを削除してメモリを解放
    del data_module, train_module, trainer


def main():
    """
    メイン関数。設定ファイルの読み込み、事前学習、学習フェーズを実行。
    """
    console = Console()

    # 並列処理の開始方法を設定
    multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_start_method("spawn", force=True)

    # 引数パーサーの設定
    arg = argparse.ArgumentParser()
    arg.add_argument("-config_path", type=str, default="./config.yml", help="学習設定ファイルのパス")
    print("◇ 設定読み込み ◇")
    print(f"{arg.parse_args().config_path}から設定を読み込みます。")

    # 設定ファイルを読み込み
    args, result_dir = load_config(config_path=arg.parse_args().config_path)

    # 再現性の確保（ランダムシードを設定）
    torch.random.manual_seed(args.others.rnd_seed)
    np.random.seed(args.others.rnd_seed)
    random.seed(args.others.rnd_seed)
    pl.seed_everything(args.others.rnd_seed, workers=True)

    # 不要な警告を無視
    warnings.filterwarnings('ignore')

    print("◇ モデル読み込み ◇")
    model = getModel()  # モデルの取得

    print("◇ TensorBoard起動 ◇")
    os.makedirs(args.others.running_path, exist_ok=True)
    tb = TensorBoard(logdir=args.others.running_path)
    tb.start()

    # 事前学習フェーズの実行
    if args.pretrain.is_valid:
        print("####################")
        print("# 事前学習フェーズ #")
        print("####################")
        print()
        print("◇ データセット読み込み ◇")
        data_dict = getPreTrainDataset()

        print("◇ データセットサンプル抽出 ◇")
        sample_dataset = data_dict.copy()["pretrain_dataset"]
        pretrain_data_sampling(
            dataset=sample_dataset,
            save_path=os.path.join(result_dir, "pretrain_data_samples")
        )

        pretrain_dataset = data_dict["pretrain_dataset"]
        num_classes_spatial, num_classes_temporal = data_dict["num_classes"]

        # 空間および時間分類ヘッドを取得
        spatial_cls_head = getPretrainClsHead(num_classes=num_classes_spatial)
        temporal_cls_head = getPretrainClsHead(num_classes=num_classes_temporal)

        # 事前学習を実行
        pretrain(
            model=model,
            spatial_cls_head=spatial_cls_head,
            temporal_cls_head=temporal_cls_head,
            dataset=pretrain_dataset,
            batch_size=args.pretrain.batch_size,
            num_workers=args.pretrain.num_workers,
            epoch=args.pretrain.epoch,
            optimizer=args.pretrain.optimizer,
            lr=args.pretrain.lr,
            weight_decay=args.pretrain.weight_decay,
            label=args.others.label,
            log_dir=args.others.running_path,
            result_dir=result_dir
        )

        # 事前学習した重みを読み込み
        model.load_state_dict(torch.load(os.path.join(result_dir,"ckpt","pretrain_checkpoint.pth")),state_dict=False)

        # メモリを解放
        gc.collect()

    # 学習フェーズの実行
    print("###################")
    print("#   学習フェーズ  #")
    print("###################")
    print()

    print("◇ データセット読み込み ◇")
    data_dict = getDataset()
    print("◇ データセットサンプル抽出 ◇")
    sample_dataset = data_dict.copy()["train_dataset"]
    data_sampling(
        dataset=sample_dataset,
        save_path=os.path.join(result_dir, "data_samples")
    )

    # 学習および検証データセットを取得
    train_dataset = data_dict["train_dataset"]
    valid_dataset = data_dict["valid_dataset"]
    num_classes = data_dict["num_classes"]

    # 分類ヘッドを取得
    cls_head = getClsHead(num_classes=num_classes)

    # モデルの学習を実行
    train(
        model=model,
        cls_head=cls_head,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=args.train.batch_size,
        num_workers=args.train.num_workers,
        epoch=args.train.epoch,
        optimizer=args.train.optimizer,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        label=args.others.label,
        log_dir=args.others.running_path,
        result_dir=result_dir,
        dataset_name=args.dataset.type  
    )



    # TensorBoardの終了
    tb.end()

    print("◇ 学習グラフ描画 ◇")
    analyzer = Analyzer(result_dir=args.others.result_path, label=args.others.label)
    with console.status("学習曲線描画中..."):
        analyzer.plot_graphs()
        analyzer.plot_cross_graphs()
    with console.status("AttentionMap描画中..."):
        analyzer.plot_attentionmap(model=model, dataset=valid_dataset)
        if args.pretrain.is_valid and os.path.join(os.path.join(result_dir, 'ckpt'), 'pretrain_checkpoint.pth'):
            analyzer.plot_attentionmap_pretrain(
                model=model, dataset=valid_dataset,
                ckpt_path=os.path.join(os.path.join(result_dir, 'ckpt'), 'pretrain_checkpoint.pth')
            )

    print("◇ 学習終了 ◇")
    print("学習結果を下記フォルダに格納しました")
    print(result_dir)


# エントリーポイント
if __name__ == '__main__':
    main()