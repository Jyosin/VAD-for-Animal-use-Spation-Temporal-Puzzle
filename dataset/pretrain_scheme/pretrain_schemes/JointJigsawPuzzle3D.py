import numpy as np
import torch
import itertools
from torchvision import transforms
from einops import rearrange
from utils.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, ToTensor

class JointJigsawPuzzle3D(object):
    """
    動画データに 3D パズル変換を適用するクラス。

    動画を空間および時間方向で分割し、ランダムなシャッフルパターンを適用します。
    一部のセルをマスクして隠すことで、事前学習タスク用のデータを生成します。

    Args:
        img_size (int): 画像サイズ（高さと幅のピクセル数）。
        n_frames (int): 動画のフレーム数。
        mask_grid_ratio (float): マスクするセルの割合 (0.0 - 1.0)。
        n_grid (tuple): 動画を分割するセル数 (高さ, 幅, 時間)。
        mean (list): 入力画像の正規化の平均値。
        std (list): 入力画像の正規化の標準偏差。
    """
    def __init__(self, img_size, n_frames, mask_grid_ratio=0.5, n_grid=(3, 3, 9), mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        # グリッド数を整数化して保存
        n_grid = [int(e) for e in list(n_grid)]
        
        # インスタンス変数の設定
        self.n_grid = n_grid  # 分割グリッド数
        self.img_size = img_size  # 画像サイズ
        self.n_frames = n_frames  # フレーム数
        
        # マスクするセル数の計算
        self.n_mask_cell = int(mask_grid_ratio * (n_grid[0] * n_grid[1] * n_grid[2]))
        
        # 全てのマスクパターンを生成
        self.whole_mask_patterns = list(itertools.combinations(np.arange(n_grid[0] * n_grid[1] * n_grid[2]), self.n_mask_cell))
        self.n_classes = len(self.whole_mask_patterns)  # マスクパターンの総数（クラス数）

        # 画像変換パイプラインの設定
        self.transforms = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(1, 1), ratio=(1, 1)),  # ランダムリサイズクロップ
            ToTensor(),  # テンソル化
            transforms.Normalize(  # 正規化
                mean=torch.tensor(mean),
                std=torch.tensor(std)
            )
        ])

    def __call__(self, x: torch.Tensor):
        """
        3D パズル変換を適用し、マスクされた動画データを生成する。

        Args:
            x (torch.Tensor): 入力動画データ（形状: (T, C, H, W)）。

        Returns:
            tuple: 
                - マスク適用後の動画データ (torch.Tensor)。
                - 対応するシャッフルパターンのインデックス (torch.Tensor)。
        """
        # 動画データに画像変換を適用
        x = self.transforms(x)  # (T, C, H, W)
        T, C, H, W = x.size()  # 動画の形状を取得
        
        # 入力サイズとグリッド分割の整合性を確認
        h, w, t = self.n_grid
        assert T % t == 0 and H % h == 0 and W % w == 0, f"Input clip size ({H}, {W}, {T}) must be divisible by n_grid ({h}, {w}, {t})."
        
        # 動画をセル単位に分割
        video_cells = rearrange(x, '(nt t) C (nh h) (nw w) -> (nt nh nw) t C h w', nh=h, nw=w, nt=t).clone()
        
        # ランダムなマスクパターンを選択
        rnd_idx = np.random.choice(self.n_classes)  # ランダムにクラスを選択
        rnd_mask_idx = list(self.whole_mask_patterns[rnd_idx])  # 対応するマスクインデックスを取得
        
        # 選択したセルをマスク（値を0に設定）
        video_cells[rnd_mask_idx] *= 0
        
        # 元の形状に再配置
        ret = rearrange(video_cells, '(nt nh nw) t C h w -> (nt t) C (nh h) (nw w)', nh=h, nw=w, nt=t)
        
        # マスク後の動画データとシャッフルパターンインデックスを返す
        return (ret, torch.Tensor([rnd_idx]).long()), (ret, torch.Tensor([rnd_idx]).long())