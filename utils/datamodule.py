import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch.utils.data.dataloader import DataLoader


class Collator(object):
    """
    学習時のデータ整形を行うためのクラス。
    通常の学習データセットにおいて、画像データとラベルをテンソルとしてバッチ化する。
    """
    def __init__(self):
        pass  # 初期化処理は不要

    def collate(self, minibatch):
        """
        ミニバッチ内のデータを整形する。
        
        Args:
            minibatch (list): 各データがタプル形式 (image, label) のリスト。

        Returns:
            list: [torch.Tensor, torch.Tensor] の形式で、1つ目は画像データのバッチ、2つ目はラベルデータのバッチ。
        """
        image_list = []
        label_list = []
        
        for record in minibatch:
            # 各レコードから画像データとラベルを抽出
            image_list.append(record[0])
            label_list.append(record[1])
        
        # データをテンソルとしてスタック
        minibatch = []
        minibatch.append(torch.stack(image_list))
        label = np.stack(label_list)
        minibatch.append(torch.from_numpy(label))  # NumPy配列をPyTorchテンソルに変換

        return minibatch


class PreTrainCollator(object):
    """
    事前学習用データを整形するためのクラス。
    空間的および時間的な分類タスクに対応するデータをバッチ化する。
    """
    def __init__(self):
        pass  # 初期化処理は不要

    def collate(self, minibatch):
        """
        データローダーに供給する際にミニバッチ内のデータを整形する。
        
        Args:
            minibatch (list): 空間および時間方向のデータが含まれるタプルリスト。

        Returns:
            list: 整形されたデータバッチ [(spatial_tensor, spatial_labels), (temporal_tensor, temporal_labels)]。
        """
        spatial_image_list = []
        spatial_label_list = []
        temporal_image_list = []
        temporal_label_list = []
        
        for spatial_data, temporal_data in minibatch:
            # 空間データをリストに追加
            spatial_image_list.append(spatial_data[0])
            spatial_label_list.append(spatial_data[1])
            # 時間データをリストに追加
            temporal_image_list.append(temporal_data[0])
            temporal_label_list.append(temporal_data[1])
        
        return [
            (torch.stack(spatial_image_list), torch.stack(spatial_label_list)),
            (torch.stack(temporal_image_list), torch.stack(temporal_label_list))
        ]


class DataModule(pl.LightningDataModule):
    """
    学習および検証データを管理するデータモジュール。

    学習用と検証用のデータローダーを生成し、ミニバッチ処理のための設定を提供する。
    """
    def __init__(
            self, 
            train_dataset: torch.utils.data.Dataset,
            valid_dataset: torch.utils.data.Dataset,
            batch_size: int = 8,
            num_workers: int = 4):
        """
        初期化メソッド。
        
        Args:
            train_dataset (torch.utils.data.Dataset): 学習データセット。
            valid_dataset (torch.utils.data.Dataset): 検証データセット。
            batch_size (int): ミニバッチサイズ。
            num_workers (int): データローダーのワーカ数。
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        """
        学習用データローダーを生成。

        Returns:
            DataLoader: 学習データを提供するデータローダー。
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=Collator().collate,  # データを整形するためにCollatorを利用
            shuffle=True,  # 学習データをシャッフル
            drop_last=True,  # 最後の不完全なバッチを無視
            pin_memory=True  # ピンメモリを利用してデータ転送を高速化
        )
    
    def val_dataloader(self):
        """
        検証用データローダーを生成。

        Returns:
            DataLoader: 検証データを提供するデータローダー。
        """
        if self.valid_dataset:
            return DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=Collator().collate,  # データを整形するためにCollatorを利用
                shuffle=False,  # 検証データの順序を保持
                drop_last=False  # 最後の不完全なバッチも使用
            )


class PreTrainDataModule(pl.LightningDataModule):
    """
    事前学習データを管理するデータモジュール。

    事前学習データのデータローダーを生成し、ミニバッチ処理のための設定を提供する。
    """
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = 8, num_workers: int = 4):
        """
        初期化メソッド。

        Args:
            dataset (torch.utils.data.Dataset): 事前学習データセット。
            batch_size (int): ミニバッチサイズ。
            num_workers (int): データローダーのワーカ数。
        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        """
        事前学習用データローダーを生成。

        Returns:
            DataLoader: 事前学習データを提供するデータローダー。
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=PreTrainCollator().collate,  # PreTrainCollatorでバッチを整形
            shuffle=True,  # データシャッフルを有効化
            drop_last=True,  # 最後の不完全なバッチを無視
            pin_memory=True  # ピンメモリを利用してデータ転送を高速化
        )