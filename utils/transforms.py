import random
import math
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# デフォルト設定
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# インターポレーションモードの変換辞書（文字列→torchのInterpolationMode、InterpolationMode→文字列）
_torch_interpolation_to_str = {
    InterpolationMode.NEAREST: 'nearest',
    InterpolationMode.BILINEAR: 'bilinear',
    InterpolationMode.BICUBIC: 'bicubic',
    InterpolationMode.BOX: 'box',
    InterpolationMode.HAMMING: 'hamming',
    InterpolationMode.LANCZOS: 'lanczos',
}
_str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}

def str_to_interp_mode(mode_str):
    """
    文字列から指定されたインターポレーションモードをtorchの`InterpolationMode`に変換する関数。

    Args:
        mode_str (str): インターポレーションモードを示す文字列。
    
    Returns:
        InterpolationMode: 対応する`InterpolationMode`オブジェクト。
    """
    return _str_to_torch_interpolation[mode_str]

class ToTensor(object):
    """
    入力画像をtorch.FloatTensorに変換し、値を[0.0, 1.0]範囲に正規化。

    Args:
        norm_value (int): 入力画像の最大値（通常は255）。
    """
    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        画像テンソルを浮動小数点数で0から1の範囲に正規化して出力。

        Args:
            pic (torch.Tensor): 入力画像のテンソル。

        Returns:
            torch.Tensor: 正規化後のテンソル。
        """
        if isinstance(pic, torch.Tensor):
            return pic.float().div(self.norm_value)

class TemporalRandomCrop(object):
    """
    動画のフレームをランダムな開始位置からクロップ。

    Args:
        size (int): クロップするフレーム数。
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        """
        動画フレームからランダムな開始位置を選択し、指定フレーム数をクロップして返す。

        Args:
            total_frames (int): 動画の全フレーム数。

        Returns:
            tuple: クロップした範囲の開始・終了インデックス。
        """
        rand_end = max(0, total_frames - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index

def getTemporalTransforms(n_frames: int, frame_interval: int):
    """
    TemporalRandomCrop を生成するファクトリ関数。

    Args:
        n_frames (int): フレーム数。
        frame_interval (int): フレーム間隔。

    Returns:
        TemporalRandomCrop: 指定サイズでのランダムクロップオブジェクト。
    """
    return TemporalRandomCrop(size=n_frames * frame_interval)

def getSpatialTransforms(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0,
    color_jitter=None,
    clop_mode="Random",
    interpolation='bilinear',
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD):
    """
    空間変換のリストを生成する関数。画像のリサイズ、反転、色調の変更、および正規化を行う。

    Args:
        img_size (int): 出力画像の目標サイズ（ピクセル単位）。
        scale (tuple): リサイズ時の範囲（高さと幅の比率）。
        ratio (tuple): リサイズ時のアスペクト比範囲。
        hflip (float): 水平方向のランダム反転の確率。
        color_jitter (float): カラージッター（色調変更）を適用する範囲。
        clop_mode (str): クロップのモード（"Random"はランダム、"Center"は中央）。
        interpolation (str): リサイズの際の補間方法。
        mean (tuple): 各チャンネルの正規化平均値。
        std (tuple): 各チャンネルの正規化標準偏差。

    Returns:
        transforms.Compose: 設定された変換を含むComposeオブジェクト。
    """
    # デフォルトスケールと比率を設定（指定がなければ (1.0, 1.0)）
    scale = tuple(scale or (1.0, 1.0))
    ratio = tuple(ratio or (1.0, 1.0))

    # 変換操作を格納するリスト
    tf = []

    # クロップモードに応じて、ランダムクロップまたは中央クロップを追加
    if clop_mode.upper() == "RANDOM":
        # ランダムにリサイズとクロップを行う変換（指定したスケールとアスペクト比で実行）
        tf = [transforms.RandomResizedCrop(img_size, scale=scale, ratio=ratio, interpolation=str_to_interp_mode(interpolation))]
    elif clop_mode.upper() == "CENTER":
        # 入力画像をリサイズし、中心部分をクロップする変換
        tf = [
            transforms.Resize(int(math.floor(img_size / DEFAULT_CROP_PCT)), interpolation=str_to_interp_mode(interpolation)),
            transforms.CenterCrop(img_size),
        ]
    
    # 水平方向の反転を指定された確率で追加
    if hflip > 0.:
        tf += [transforms.RandomHorizontalFlip(p=hflip)]

    # カラージッター（明るさ、コントラスト、彩度、色相の変化）を追加
    if color_jitter is not None:
        # color_jitterに指定した値を使用して変換を追加
        color_jitter = (float(color_jitter),) * 3  # 各パラメータ（明るさ、コントラスト、彩度）に同じ値を適用
        tf += [transforms.ColorJitter(*color_jitter)]

    # テンソルへの変換と、標準化（正規化）変換を追加
    tf += [
        ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ]

    # 上記の変換リストをComposeオブジェクトでラップし、最終的な変換を返す
    return transforms.Compose(tf)