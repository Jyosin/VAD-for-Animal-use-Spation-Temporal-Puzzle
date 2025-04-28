import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchmetrics import Accuracy
# from utils.utils import print


class TrainModule(pl.LightningModule):
    """
    モデルの学習用モジュール。

    モデル、分類ヘッド、最適化関数、学習率、正則化の設定を行い、学習ステップや検証ステップ、
    エポック終了時の処理を定義する。

    Args:
        model (torch.nn.Module): 学習対象のモデル。
        cls_head (torch.nn.Module): 分類用のヘッド。
        optimizer (str): 最適化手法の名前（例: "SGD", "AdamW", "Adam"）。
        lr (float): 学習率。
        weight_decay (float): 正則化の重み（Weight Decay）。
        result_dir (str): 結果保存ディレクトリ。
    """
    def __init__(self, model, cls_head, optimizer, lr, weight_decay, result_dir, dataset_name=None):

        super().__init__()

        # モデルと分類ヘッドの設定
        self.model = model
        self.cls_head = cls_head

        # オプティマイザ設定
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay

        # 分類クラス数の取得
        self.n_class = cls_head.num_classes

        # 損失関数（クロスエントロピー）
        self.loss_fn = nn.CrossEntropyLoss()
        self.dataset_name = dataset_name

        # ディレクトリ設定
        self.result_dir = result_dir
        self.ckpt_dir = os.path.join(result_dir, "ckpt")  # チェックポイントの保存先
        os.makedirs(self.ckpt_dir, exist_ok=True)

    # データセット名に応じてtop_kを設定
        if self.dataset_name is not None and "animalkingdom" in self.dataset_name.lower():
            # 動物データセットならtop_k=1
            top_k = 1
        else:
            top_k = 5

        # トップ1精度計測器（学習中・検証中）
        self.train_top1_acc = Accuracy(task="multiclass", num_classes=self.n_class, top_k=1)
        self.val_top1_acc = Accuracy(task="multiclass", num_classes=self.n_class, top_k=1)

        # トップk精度計測器（学習中・検証中）
        self.train_topk_acc = Accuracy(task="multiclass", num_classes=self.n_class, top_k=top_k)
        self.val_topk_acc = Accuracy(task="multiclass", num_classes=self.n_class, top_k=top_k)
       

        # 学習と検証ログを記録するディクショナリ
        self.learning_log = {
            "train": {
                "loss": {},
                "top1_acc": {},
                "top5_acc": {}
            },
            "valid": {
                "loss": {},
                "top1_acc": {},
                "top5_acc": {}
            }
        }

    def configure_optimizers(self):
        """
        最適化関数を構築して返すメソッド。

        Returns:
            list: 構築された最適化関数のリスト。
        """
        opt = self.optimizer.upper()
        lr = self.lr
        weight_decay = self.weight_decay

        # 最適化手法を選択
        optimizer = optim.SGD(self.model.parameters(), momentum=0.9, nesterov=True, lr=lr, weight_decay=weight_decay)
        if opt == 'ADAMW':
            optimizer = optim.AdamW(self.model.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=weight_decay)
        elif opt == 'ADAM':
            optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=weight_decay)

        return [optimizer]

    def on_train_epoch_start(self):
        """
        各エポックの開始時にログを初期化するメソッド。
        """
        self.learning_log["train"]["loss"][str(self.current_epoch)] = []
        self.learning_log["train"]["top1_acc"][str(self.current_epoch)] = []
        self.learning_log["train"]["top5_acc"][str(self.current_epoch)] = []

    def training_step(self, batch, batch_idx):
        """
        学習ステップの処理を定義するメソッド。

        Args:
            batch (tuple): バッチデータ（入力データとラベル）。
            batch_idx (int): バッチインデックス。

        Returns:
            dict: 計算されたロスを含む辞書。
        """
        inputs, labels = batch

        # モデル出力を取得
        preds = self.model(inputs)
        preds = self.cls_head(preds)

        # ロス計算
        cls_loss = self.loss_fn(preds, labels)
        loss = cls_loss

        # 精度計算
        top1_acc = self.train_top1_acc(preds.softmax(dim=-1), labels)
        top5_acc = self.train_top5_acc(preds.softmax(dim=-1), labels)

        # ログ記録
        self.learning_log["train"]["loss"][str(self.current_epoch)].append(loss.item())
        self.learning_log["train"]["top1_acc"][str(self.current_epoch)].append(top1_acc.item())
        self.learning_log["train"]["top5_acc"][str(self.current_epoch)].append(top5_acc.item())

        # ログの保存
        self.log("loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("top1_acc", top1_acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("top5_acc", top5_acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {'loss': loss}

    def on_train_epoch_end(self):
        """
        各エポックの終了時に実行される処理。
        エポックごとの精度を表示し、精度計測器をリセット。
        """
        mean_top1_acc = self.train_top1_acc.compute()
        mean_top5_acc = self.train_top5_acc.compute()

        # 精度をログ出力
        print(f'EPOCH[{self.trainer.current_epoch:02d}/{self.trainer.max_epochs:02d}] - TRAIN ACCURACY -> top1_acc: {mean_top1_acc:.3f}, top5_acc: {mean_top5_acc:.3f}')

        # 計測器をリセット
        self.train_top1_acc.reset()
        self.train_top5_acc.reset()

        # チェックポイントの保存
        save_path = os.path.join(self.ckpt_dir, f'checkpoint-{self.current_epoch:02d}epoch.pth')
        self.trainer.save_checkpoint(save_path)

    def on_validation_epoch_start(self):
        """
        各エポックの検証開始時にログを初期化するメソッド。
        """
        self.learning_log["valid"]["loss"][str(self.current_epoch)] = []
        self.learning_log["valid"]["top1_acc"][str(self.current_epoch)] = []
        self.learning_log["valid"]["top5_acc"][str(self.current_epoch)] = []

    def validation_step(self, batch):
        """
        検証ステップを定義するメソッド。

        Args:
            batch (tuple): バッチデータ（入力データとラベル）。
        """
        inputs, labels = batch
        with torch.no_grad():
            # モデル出力を取得
            preds = self.model(inputs)
            preds = self.cls_head(preds)

            # ロス計算
            loss = self.loss_fn(preds, labels)

        # 精度計算
        top1_acc = self.val_top1_acc(preds.softmax(dim=-1), labels)
        top5_acc = self.val_top5_acc(preds.softmax(dim=-1), labels)

        # ログ記録
        self.learning_log["valid"]["loss"][str(self.current_epoch)].append(loss.item())
        self.learning_log["valid"]["top1_acc"][str(self.current_epoch)].append(top1_acc.item())
        self.learning_log["valid"]["top5_acc"][str(self.current_epoch)].append(top5_acc.item())

    def on_validation_epoch_end(self):
        """
        検証エポック終了時に実行される処理。
        """
        mean_top1_acc = self.val_top1_acc.compute()
        mean_top5_acc = self.val_top5_acc.compute()

        # 精度をログ出力
        print(f'EPOCH[{self.trainer.current_epoch:02d}/{self.trainer.max_epochs:02d}] - VALID ACCURACY -> top1_acc: {mean_top1_acc:.3f}, top5_acc: {mean_top5_acc:.3f}')

        # 計測器をリセット
        self.val_top1_acc.reset()
        self.val_top5_acc.reset()

    def on_train_end(self):
        """
        学習終了時に学習ログを保存する。
        """
        import pickle
        with open(os.path.join(self.result_dir, "learning_log.pkl"), mode='wb') as f:
            pickle.dump(self.learning_log, f)