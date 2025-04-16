import numpy as np
import torch
import itertools
from torchvision import transforms
from einops import rearrange
from utils.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, ToTensor


class SeparateJigsawPuzzle3D(object):
    """
    動画データに 3D パズル変換を適用するクラス。

    動画を空間方向と時間方向に分割し、それぞれのグリッドでランダムなシャッフルパターンを適用します。
    空間と時間のマスクを分離することで、2つの異なる予測タスクを生成します。

    Args:
        img_size (int): 画像サイズ（高さと幅のピクセル数）。
        n_frames (int): 動画のフレーム数。
        mask_grid_ratio (tuple): 空間方向と時間方向のマスクするセルの割合 (0.0 - 1.0)。
        n_grid (tuple): 動画を分割するセル数 (高さ, 幅, 時間)。
        mean (list): 入力画像の正規化の平均値。
        std (list): 入力画像の正規化の標準偏差。
    """
    def __init__(self, img_size, n_frames, mask_grid_ratio=(0.5, 0.5), n_grid=(3, 3, 9), mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        # グリッド数を整数化して保存
        n_grid = [int(e) for e in list(n_grid)]
        
        # インスタンス変数の設定
        self.n_grid = n_grid  # 分割グリッド数
        self.img_size = img_size  # 画像サイズ
        self.n_frames = n_frames  # フレーム数
        
        # 空間および時間方向のマスクするセル数の計算
        self.n_spaltial_mask_grid = int(mask_grid_ratio[0] * (n_grid[0] * n_grid[1]))
        self.n_temporal_mask_grid = int(mask_grid_ratio[1] * n_grid[2])

        # 空間および時間の全マスクパターンを生成
        self.whole_spatial_mask_patterns = list(itertools.combinations(np.arange(n_grid[0] * n_grid[1]), self.n_spaltial_mask_grid))
        self.n_classes_spatial = len(self.whole_spatial_mask_patterns)  # 空間方向のクラス数

        self.whole_temporal_mask_patterns = list(itertools.combinations(np.arange(n_grid[2]), self.n_temporal_mask_grid))
        self.n_classes_temporal = len(self.whole_temporal_mask_patterns)  # 時間方向のクラス数

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
        空間方向と時間方向に分割した 3D パズル変換を適用し、マスクされた動画データを生成します。

        Args:
            x (torch.Tensor): 入力動画データ（形状: (T, C, H, W)）。

        Returns:
            tuple:
                - 空間方向のマスク適用後の動画データと対応するシャッフルパターンインデックス。
                - 時間方向のマスク適用後の動画データと対応するシャッフルパターンインデックス。
        """
        # 動画データに画像変換を適用
        x = self.transforms(x)  # (T, C, H, W)
        T, C, H, W = x.size()  # 動画の形状を取得
        
        # 入力サイズとグリッド分割の整合性を確認
        h, w, t = self.n_grid
        assert T % t == 0 and H % h == 0 and W % w == 0, f"Input clip size ({H}, {W}, {T}) must be divisible by n_grid ({h}, {w}, {t})."
        
        # 空間方向と時間方向でグリッド分割
        spatial_cells = rearrange(x, 'T C (nh h) (nw w) -> T C (nh nw) h w', nh=h, nw=w).clone()
        temporal_cells = rearrange(x, '(nt t) C H W -> nt t C H W', nt=t).clone()
        
        # ランダムに空間と時間のマスクパターンを選択
        rnd_idx_t = np.random.choice(self.n_classes_temporal)  # 時間方向のクラスインデックス
        rnd_idx_hw = np.random.choice(self.n_classes_spatial)  # 空間方向のクラスインデックス

        rnd_mask_idx_t = list(self.whole_temporal_mask_patterns[rnd_idx_t])  # 時間方向のマスクインデックス
        rnd_mask_idx_hw = list(self.whole_spatial_mask_patterns[rnd_idx_hw])  # 空間方向のマスクインデックス
        
        # 空間方向のセルをマスク
        spatial_cells[:, :, rnd_mask_idx_hw] *= 0
        # 時間方向のセルをマスク
        temporal_cells[rnd_mask_idx_t] *= 0

        # マスク後のセルを再構成
        spatial_ret = rearrange(spatial_cells, 'T C (nh nw) h w -> T C (nh h) (nw w)', nh=h, nw=w)
        temporal_ret = rearrange(temporal_cells, 'nt t C H W -> (nt t) C H W', nt=t)

        # 空間および時間方向のマスク結果と対応インデックスを返す
        return (spatial_ret, torch.Tensor([rnd_idx_hw]).long()), (temporal_ret, torch.Tensor([rnd_idx_t]).long())