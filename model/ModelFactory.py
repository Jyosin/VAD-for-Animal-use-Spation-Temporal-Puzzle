import torch
import os, sys

# カレントディレクトリをパスに追加
sys.path.append(os.getcwd())

# 必要なモジュールのインポート
from utils.utils import load_yml
from model.models.SwinTransformer import SwinTransformer3D
from model.models.C3D import C3D
from model.models.TimeSformer import TimeSformer
from model.models.R3D import ResNet
from model.models.ClassificationHeads import SwinHead, C3DHead, TimeSFormerHead, R3DHead
from model.models.ClassificationHeads import SwinPretrainHead, C3DPretrainHead, TimeSFormerPretrainHead, R3DPretrainHead


# グローバル変数でモデルのパラメータ設定を保持
model_params = None

# モデル定義ファイルのパスを設定
yml_path = os.path.join(os.path.dirname(__file__), "model_zoo.yml")
model_zoo = load_yml(yml_path)  # モデル定義ファイルを読み込み

def initModelConfig(model_type: str):
    """
    指定されたモデルタイプの設定を初期化する関数。指定されたモデルタイプに基づき、
    model_params に対応する設定をロードする。

    Args:
        model_type (str): モデルの種類（例："SwinTransformer3D"）。

    Raises:
        Exception: モデルタイプがモデル定義ファイルに存在しない場合。
    """
    # 指定されたモデルタイプが定義されているか確認
    if not hasattr(model_zoo, model_type):
        raise Exception(f"モデル定義ファイル({os.path.join(os.path.dirname(__file__), 'model_zoo.yml')})にタイプ{model_type}が定義されていません。")

    global model_params
    model_params = model_zoo.__getattribute__(model_type)  # 指定したモデルタイプの設定を取得

def getModel() -> torch.nn.Module:
    """
    指定されたモデルタイプに応じて、モデルのインスタンスを生成して返す関数。

    Returns:
        torch.nn.Module: 指定されたモデルタイプに基づくモデルのインスタンス。
   
    Raises:
        Exception: 指定したモデルタイプがモデル定義ファイルに存在しない場合。
    """
    # SwinTransformer3D モデルを使用する場合
    if model_params.Architecture.upper() == "SWINTRANSFORMER3D":
        print("モデルタイプ：SwinTransformer3D")
        model = SwinTransformer3D(
            patch_size=model_params.patch_size,
            embed_dim=model_params.embed_dim,
            depths=model_params.depths,
            num_heads=model_params.num_heads,
            window_size=model_params.window_size,
            mlp_ratio=model_params.mlp_ratio,
            qkv_bias=model_params.qkv_bias,
            qk_scale=model_params.qk_scale,
            drop_rate=model_params.drop_rate,
            attn_drop_rate=model_params.attn_drop_rate,
            drop_path_rate=model_params.drop_path_rate,
            patch_norm=model_params.patch_norm
        )
    
    # C3D モデルを使用する場合
    elif model_params.Architecture.upper() == "C3D":
        print("モデルタイプ：C3D")
        model = C3D()

    # TimeSFormer モデルを使用する場合
    elif model_params.Architecture.upper() == "TIMESFORMER":
        print("モデルタイプ：TimeSFormer")
        from dataset.DatasetFactory import dataset_params
        model = TimeSformer(
            img_size=dataset_params.img_size,
            patch_size=model_params.patch_size,
            num_frames=dataset_params.n_frames,
            attention_type=model_params.attention_type,
            embed_dim=model_params.embed_dim,
            depth=model_params.depth,
            num_heads=model_params.num_heads,
            mlp_ratio=model_params.mlp_ratio,
            qkv_bias=model_params.qkv_bias,
            drop_rate=model_params.drop_rate,
            attn_drop_rate=model_params.attn_drop_rate,
            drop_path_rate=model_params.drop_path_rate
        )

    # 3D-Resnet モデルを使用する場合
    elif model_params.Architecture.upper() == "3D-RESNET":
        print("モデルタイプ：3D-Resnet")
        model = ResNet(
            layers=model_params.layers
        )

    # 作成したモデルのインスタンスを返す
    return model

def getClsHead(num_classes: int) -> torch.nn.Module:
    """
    指定されたモデルタイプに基づき、分類用のヘッドを取得する関数。

    Args:
        num_classes (int): 分類するクラス数。

    Returns:
        torch.nn.Module: 指定されたモデルタイプに基づく分類用ヘッドのインスタンス。
    
    Raises:
        Exception: 指定したモデルタイプがモデル定義ファイルに存在しない場合。
    """
    # SwinTransformer3D モデルの分類ヘッドを使用する場合
    if model_params.Architecture.upper() == "SWINTRANSFORMER3D":
        head = SwinHead(
            num_classes=num_classes,
            in_channels=model_params.cls_head.in_channels
        )
    
    # C3D モデルの分類ヘッドを使用する場合
    elif model_params.Architecture.upper() == "C3D":
        head = C3DHead(
            num_classes=num_classes,
            in_channels=model_params.cls_head.in_channels,
        )

    # TimeSFormer モデルの分類ヘッドを使用する場合
    elif model_params.Architecture.upper() == "TIMESFORMER":
        head = TimeSFormerHead(
            num_classes=num_classes,
            in_channels=model_params.cls_head.in_channels
        )

    # 3D-Resnet モデルの分類ヘッドを使用する場合
    elif model_params.Architecture.upper() == "3D-RESNET":
        head = R3DHead(
            num_classes=num_classes,
            in_channels=model_params.cls_head.in_channels
        )

    # 作成した分類ヘッドのインスタンスを返す
    return head

def getPretrainClsHead(num_classes: int) -> torch.nn.Module:
    """
    指定されたモデルタイプに基づき、分類用のヘッドを取得する関数。

    Args:
        num_classes (int): 分類するクラス数。

    Returns:
        torch.nn.Module: 指定されたモデルタイプに基づく分類用ヘッドのインスタンス。
    
    Raises:
        Exception: 指定したモデルタイプがモデル定義ファイルに存在しない場合。
    """
    from dataset.DatasetFactory import pretrain_scheme_params
    if pretrain_scheme_params.type.upper()=="CUBICPUZZLE3D":
        in_channels=int(model_params.pretrain_cls_head.in_channels*pretrain_scheme_params.n_grid[2])
        n_cells=int(pretrain_scheme_params.n_grid[2])
    elif pretrain_scheme_params.type.upper()=="SEPARATEJIGSAWPUZZLE3D":
        in_channels=model_params.pretrain_cls_head.in_channels
        n_cells=None
    elif pretrain_scheme_params.type.upper()=="JOINTJIGSAWPUZZLE3D":
        in_channels=model_params.pretrain_cls_head.in_channels
        n_cells=None
        
    # SwinTransformer3D モデルの分類ヘッドを使用する場合
    if model_params.Architecture.upper() == "SWINTRANSFORMER3D":
        head = SwinPretrainHead(
            num_classes=num_classes,
            in_channels=in_channels,
            n_cells=n_cells,
            pretrain_scheme=pretrain_scheme_params.type
        )
    # C3D モデルの分類ヘッドを使用する場合
    elif model_params.Architecture.upper() == "C3D":
        head = C3DPretrainHead(
            num_classes=num_classes,
            in_channels=in_channels,
            n_cells=n_cells,
            pretrain_scheme=pretrain_scheme_params.type
        )
    # TimeSFormer モデルの分類ヘッドを使用する場合
    elif model_params.Architecture.upper() == "TIMESFORMER":
        head = TimeSFormerPretrainHead(
            num_classes=num_classes,
            in_channels=in_channels,
            n_cells=n_cells,
            pretrain_scheme=pretrain_scheme_params.type
        )
    # 3D-Resnet モデルの分類ヘッドを使用する場合
    elif model_params.Architecture.upper() == "3D-RESNET":
        head = R3DPretrainHead(
            num_classes=num_classes,
            in_channels=in_channels,
            n_cells=n_cells,
            pretrain_scheme=pretrain_scheme_params.type
        )

    # 作成した分類ヘッドのインスタンスを返す
    return head

# メイン関数
if __name__ == '__main__':
    # モデルの取得テスト
    getModel("Swin-S")