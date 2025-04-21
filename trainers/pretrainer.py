import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from einops import rearrange


class PreTrainModule(pl.LightningModule):
    """
    事前学習用のモジュール。

    モデル、分類ヘッド、最適化関数、学習率、正則化の設定を行い、
    空間的および時間的分類ヘッドを用いて学習ステップとエポック終了時の処理を定義します。

    Args:
        model (torch.nn.Module): ベースモデル。
        spatial_cls_head (torch.nn.Module): 空間的分類ヘッド。
        temporal_cls_head (torch.nn.Module): 時間的分類ヘッド。
        optimizer (str): 最適化手法の名前（"SGD", "AdamW", "Adam"）。
        lr (float): 学習率。
        weight_decay (float): 正則化の重み（Weight Decay）。
        result_dir (str): 結果保存ディレクトリ。
    """
    def __init__(self, model, spatial_cls_head, temporal_cls_head, optimizer, lr, weight_decay, result_dir):
        super().__init__()
        from dataset.DatasetFactory import pretrain_scheme_params
        self.pretrain_scheme_type = pretrain_scheme_params.type

        # モデル設定
        self.model = model
        self.model._pretrain()  # モデルを事前学習モードに設定

        # 空間的および時間的分類ヘッド
        self.spatial_cls_head = spatial_cls_head
        self.temporal_cls_head = temporal_cls_head

        # 最適化設定
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay

        # クラス数を取得
        self.n_class_spatial = spatial_cls_head.num_classes
        self.n_class_temporal = temporal_cls_head.num_classes

        # 精度計測
        self.train_top1_acc_spatial = Accuracy(task="multiclass", num_classes=self.n_class_spatial)
        self.train_top1_acc_temporal = Accuracy(task="multiclass", num_classes=self.n_class_temporal)

        # 損失関数
        self.loss_fn = nn.CrossEntropyLoss()

        # ディレクトリ設定
        self.result_dir = result_dir
        self.ckpt_dir = os.path.join(result_dir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 学習ログの初期化
        self.learning_log = {
            "train": {
                "loss": {},
                "spatial_loss": {},
                "temporal_loss": {},
                "spatial_top1_acc": {},
                "temporal_top1_acc": {}
            }
        }

    def configure_optimizers(self):
        """
        最適化関数を設定する。

        Returns:
            list: 構築した最適化関数のリスト。
        """
        opt = self.optimizer.upper()
        lr = self.lr
        weight_decay = self.weight_decay

        # 最適化関数を選択
        optimizer = optim.SGD(self.model.parameters(), momentum=0.9, nesterov=True, lr=lr, weight_decay=weight_decay)
        if opt == 'ADAMW':
            optimizer = optim.AdamW(self.model.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=weight_decay)
        elif opt == 'ADAM':
            optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=weight_decay)

        return [optimizer]

    def on_train_epoch_start(self):
        """
        各エポックの開始時にログを初期化します。
        """
        self.learning_log["train"]["loss"][str(self.current_epoch)] = []
        self.learning_log["train"]["spatial_loss"][str(self.current_epoch)] = []
        self.learning_log["train"]["temporal_loss"][str(self.current_epoch)] = []
        self.learning_log["train"]["spatial_top1_acc"][str(self.current_epoch)] = []
        self.learning_log["train"]["temporal_top1_acc"][str(self.current_epoch)] = []

    def pretrain_transform(self, tensor: torch.Tensor):
        """
        入力データを事前学習用に整形します。

        Args:
            tensor (torch.Tensor): 入力テンソル。

        Returns:
            torch.Tensor: 整形後のテンソル。
        """
        if self.pretrain_scheme_type.upper() == "CUBICPUZZLE3D":
            return rearrange(tensor, 'B N T C H W -> (B N) T C H W')
        else:
            return tensor

    def training_step(self, batch):
        """
        学習ステップを定義します。

        Args:
            batch (tuple): バッチデータ（空間的および時間的なパズルデータ）。

        Returns:
            dict: 計算されたロス。
        """
        spatial_puzzle, temporal_puzzle = batch[0], batch[1]

        # 空間的パズル処理
        spatial_inputs, spatial_labels = spatial_puzzle
        spatial_inputs = self.pretrain_transform(spatial_inputs)
        spatial_preds = self.model(spatial_inputs)
        spatial_preds = self.spatial_cls_head(spatial_preds)
        spatial_loss = self.loss_fn(spatial_preds, spatial_labels.flatten())

        # 時間的パズル処理
        if self.pretrain_scheme_type.upper() == "SEPARATEJIGSAWPUZZLE3D":
            temporal_inputs, temporal_labels = temporal_puzzle
            temporal_inputs = self.pretrain_transform(temporal_inputs)
            temporal_preds = self.model(temporal_inputs)
            temporal_preds = self.temporal_cls_head(temporal_preds)
            temporal_loss = self.loss_fn(temporal_preds, temporal_labels.flatten())
        else:
            temporal_loss = 0

        # 総損失
        loss = spatial_loss + temporal_loss

        # トップ1精度の計算
        top1_spatial_acc = self.train_top1_acc_spatial(spatial_preds.softmax(dim=-1), spatial_labels.flatten())
        self.learning_log["train"]["loss"][str(self.current_epoch)].append(loss.item())
        self.learning_log["train"]["spatial_loss"][str(self.current_epoch)].append(spatial_loss.item())
        self.learning_log["train"]["spatial_top1_acc"][str(self.current_epoch)].append(top1_spatial_acc.item())

        # ログに記録
        self.log("spatial_loss", spatial_loss, on_step=True, prog_bar=True)
        self.log("top1_spatial_acc", top1_spatial_acc, on_step=True, prog_bar=True)

        if self.pretrain_scheme_type.upper() == "SEPARATEJIGSAWPUZZLE3D":
            top1_temporal_acc = self.train_top1_acc_temporal(temporal_preds.softmax(dim=-1), temporal_labels.flatten())
            self.learning_log["train"]["temporal_loss"][str(self.current_epoch)].append(temporal_loss.item())
            self.learning_log["train"]["temporal_top1_acc"][str(self.current_epoch)].append(top1_temporal_acc.item())
            self.log("temporal_loss", temporal_loss, on_step=True, prog_bar=True)
            self.log("top1_temporal_acc", top1_temporal_acc, on_step=True, prog_bar=True)

        return {'loss': loss}

    def on_train_epoch_end(self):
        """
        各エポック終了時に実行される処理。
        精度計測器をリセットします。
        """
        self.train_top1_acc_spatial.reset()
        if self.pretrain_scheme_type.upper() == "SEPARATEJIGSAWPUZZLE3D":
            self.train_top1_acc_temporal.reset()

    def on_train_end(self):
        """
        学習終了時にログとチェックポイントを保存します。
        """
        import pickle
        with open(os.path.join(self.result_dir, "pretrain_learning_log.pkl"), mode='wb') as f:
            pickle.dump(self.learning_log, f)
        save_path = os.path.join(self.ckpt_dir, 'pretrain_checkpoint.pth')
        self.trainer.save_checkpoint(save_path)