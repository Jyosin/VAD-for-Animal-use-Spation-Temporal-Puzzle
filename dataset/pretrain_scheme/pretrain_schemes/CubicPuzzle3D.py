import random
import itertools
import numpy as np
import torch

from torchvision import transforms
from einops import rearrange
from utils.transforms import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD,ToTensor


class CubicPuzzle3D(object):
    """
    3Dのキュービックパズルを生成し、動画データを空間と時間軸でランダムにシャッフルするクラス。
    動画をグリッドに分割し、空間および時間的なシャッフルを実行。シャッフル後のインデックスを含む。

    Args:
        img_size (int): 入力画像サイズ。
        n_frames (int): 動画のフレーム数。
        n_grid (tuple): 分割するグリッドサイズ（高さ、幅、時間のセル数を示すタプル）。
        grayscale_prob (float): グレースケール変換を適用する確率。
        jitter_size (tuple): 各セルの高さ、幅、時間方向のジッターサイズ。
        mean (tuple): 正規化の平均値。
        std (tuple): 正規化の標準偏差。
    """
    def __init__(self, img_size, n_frames, n_grid=(2,2,4), grayscale_prob=0.5, jitter_size=(80,80,16), mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        n_grid = [int(e) for e in list(n_grid)]
        jitter_size = [int(e) for e in list(jitter_size)]
        
        # n_gridの高さ・幅がセル数に一致するか確認
        assert n_grid[0] * n_grid[1] == n_grid[2], f"n_grid (h,w,t) must be h*w=t, current n_grid (h,w,t) = ({n_grid[0]},{n_grid[1]},{n_grid[2]}) do not satisfy"
        
        # パラメータの保存とシャッフルパターンの設定
        self.n_grid = n_grid
        self.img_size = img_size
        self.n_frames = n_frames
        self.grayscale_prob = grayscale_prob
        self.jitter_size = jitter_size
        
        # 空間・時間方向のシャッフルパターンを事前に生成
        self.spatial_shuffle_patterns = list(itertools.permutations(np.arange(int(n_grid[0] * n_grid[1]))))
        self.temporal_shuffle_patterns = list(itertools.permutations(np.arange(int(n_grid[2]))))
        
        # グレースケール変換と画像リサイズ用のランダムリサイズクロップ
        self.grayscale = transforms.Grayscale(3)
        resize_fn = transforms.RandomResizedCrop(img_size, scale=(1,1), ratio=(1,1))

        # 一連の画像変換を定義
        self.transforms = transforms.Compose([
            resize_fn,
            ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std)
            )
        ])

    def __call__(self, x: torch.Tensor):
        """
        入力テンソルに対して変換を適用し、空間・時間方向のランダムなシャッフル後のセルを取得。

        Args:
            x (torch.Tensor): 入力動画データ（形状：T, C, H, W）。

        Returns:
            tuple: 空間および時間シャッフルされたセルのテンソルと、シャッフルインデックス。
        """
        # グレースケール変換の確率適用
        if random.uniform(0, 1) > self.grayscale_prob:
            x = self.grayscale(x)
        
        # 画像変換を適用
        x = self.transforms(x)
        T, C, H, W = x.size()
        
        # 入力サイズとセル分割の整合性を確認
        h, w, t = self.n_grid
        assert T % t == 0 and H % h == 0 and W % w == 0, f"Input clip size ({H},{W},{T}) must be dividable by n_grid ({h},{w},{t})"
        
        # 動画をセルに分割（空間・時間セルを生成）
        cells = rearrange(x, '(nt t) C (nh h) (nw w) -> nt (nh nw) t C h w ', nh=h, nw=w, nt=t)

        # 時間と空間のセルをランダムに選択
        rnd_select_t = np.random.choice(t)
        rnd_select_hw = np.random.choice(int(h * w))
        spatial_cells = cells[rnd_select_t]
        temporal_cells = cells[:, rnd_select_hw]
        
        # ランダムにシャッフルパターンを選択して適用
        rnd_shuffle_idx_t = np.random.choice(len(self.temporal_shuffle_patterns))
        rnd_shuffle_t = list(self.temporal_shuffle_patterns[rnd_shuffle_idx_t])
        rnd_shuffle_idx_hw = np.random.choice(len(self.spatial_shuffle_patterns))
        rnd_shuffle_hw = list(self.spatial_shuffle_patterns[rnd_shuffle_idx_hw])

        spatial_cells = spatial_cells[rnd_shuffle_hw]
        temporal_cells = temporal_cells[rnd_shuffle_t]
        
        # セルごとにジッターを追加
        nh, nw, nt = H // h, W // w, T // t
        jitter_size_h, jitter_size_w, jitter_size_t = self.jitter_size
        assert nh >= jitter_size_h and nw >= jitter_size_w and nt >= jitter_size_t, f"Jitter size ({jitter_size_h},{jitter_size_w},{jitter_size_t}) must be smaller than cell size ({nh},{nw},{nt})"
        
        # 空間セルのジッター後の出力テンソルを作成
        spatial_ret = torch.zeros(int(h * w), jitter_size_t, 3, jitter_size_h, jitter_size_w)
        for i in range(spatial_cells.size(0)):
            h_jitter_idx = np.random.choice(nh - jitter_size_h) if nh > jitter_size_h else 0
            w_jitter_idx = np.random.choice(nw - jitter_size_w) if nw > jitter_size_w else 0
            t_jitter_idx = np.random.choice(nt - jitter_size_t) if nt > jitter_size_t else 0
            spatial_ret[i] = spatial_cells[i, t_jitter_idx:t_jitter_idx+jitter_size_t, :, h_jitter_idx:h_jitter_idx+jitter_size_h, w_jitter_idx:w_jitter_idx+jitter_size_w]

        # 時間セルのジッター後の出力テンソルを作成
        temporal_ret = torch.zeros(t, jitter_size_t, 3, jitter_size_h, jitter_size_w)
        for i in range(temporal_cells.size(0)):
            h_jitter_idx = np.random.choice(nh - jitter_size_h) if nh > jitter_size_h else 0
            w_jitter_idx = np.random.choice(nw - jitter_size_w) if nw > jitter_size_w else 0
            t_jitter_idx = np.random.choice(nt - jitter_size_t) if nt > jitter_size_t else 0
            temporal_ret[i] = temporal_cells[i, t_jitter_idx:t_jitter_idx+jitter_size_t, :, h_jitter_idx:h_jitter_idx+jitter_size_h, w_jitter_idx:w_jitter_idx+jitter_size_w]

        return (spatial_ret, torch.Tensor([rnd_shuffle_idx_hw]).long()), (temporal_ret, torch.Tensor([rnd_shuffle_idx_t]).long())