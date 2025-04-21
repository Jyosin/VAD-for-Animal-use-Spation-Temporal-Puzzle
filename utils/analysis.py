import pickle
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import glob, os
from model.models.TimeSformer import TimeSformer
from model.models.SwinTransformer import SwinTransformer3D
from einops import rearrange

class Analyzer:
    """
    学習結果のログから損失と精度の変遷をプロットし、Attention Mapを生成するためのクラス。

    Attributes:
        result_dir (str): 学習結果が保存されているディレクトリへのパス。
        results (dict): 学習ログを格納する辞書。ディレクトリから読み込まれる。
        color_map (Colormap): プロット時に使用されるカラーマップ。
        attn_modelType (list): 対応するモデルタイプ（TimeSformer, SwinTransformer3D）。
    """
    color_map = plt.get_cmap("tab10")  # カラーマップを設定
    attn_modelType = [TimeSformer, SwinTransformer3D]

    def __init__(self, result_dir: str, label: str):
        """
        Analyzer クラスのコンストラクタ。

        Args:
            result_dir (str): 結果ディレクトリへのパス。ディレクトリ内の学習ログを読み込む。
            label (str): 実験ラベル。
        """
        self.result_dir = result_dir
        self.results = {}
        self.label = label

        # 指定されたディレクトリ内の学習結果を読み込む
        for result_path in glob.glob(result_dir + "/*"):
            if not os.path.exists(os.path.join(result_path, "learning_log.pkl")):
                continue
            with open(os.path.join(result_path, "learning_log.pkl"), mode="rb") as f:
                self.results[os.path.basename(result_path)] = pickle.load(f)

    def plot_graphs(self, targets: list = [], train_color: str = "blue", valid_color: str = "orange", plot_whole_data: bool = False):
        """
        各実験の損失と精度をプロットし、保存する。

        Args:
            targets (list): プロット対象とする実験の名前リスト。空の場合、全てプロットする。
            train_color (str): 訓練データ損失のプロット色。
            valid_color (str): 検証データ損失のプロット色。
            plot_whole_data (bool): 全データをプロットするかのフラグ。
        """
        if len(targets) == 0: 
            targets = self.results.keys()

        # 各実験について損失と精度をプロット
        for tag, logs in self.results.items():
            if tag not in targets:
                continue

            # 損失のプロット
            plt.figure(figsize=(6, 4), layout="tight")
            plt.xlabel("epoch")
            plt.ylabel("loss")

            # 訓練データの損失
            print_x, print_losses = np.empty(0), np.empty(0)
            for epoch, losses in logs["train"]["loss"].items():
                epoch = int(epoch)
                print_x = np.concatenate([print_x, np.linspace(start=epoch, stop=epoch+1, num=len(losses))])
                print_losses = np.concatenate([print_losses, losses])
            moving_avg = np.convolve(print_losses, np.ones(int(len(print_losses)/100))/(len(print_losses)/100), mode="same")
            plt.plot(print_x, moving_avg, color=train_color, linestyle="-", label="train", lw=1)
            if plot_whole_data:
                plt.plot(print_x, print_losses, color=train_color, linestyle="-", alpha=0.3, lw=0.5)

            # 検証データの損失
            print_x, print_losses = np.empty(0), np.empty(0)
            for epoch, losses in logs["valid"]["loss"].items():
                epoch = int(epoch)
                print_x = np.concatenate([print_x, np.linspace(start=epoch, stop=epoch+1, num=len(losses))])
                print_losses = np.concatenate([print_losses, losses])
            moving_avg = np.convolve(print_losses, np.ones(int(len(print_losses)/100))/(len(print_losses)/100), mode="same")
            plt.plot(print_x, moving_avg, color=valid_color, linestyle="-", label="valid", lw=1)
            if plot_whole_data:
                plt.plot(print_x, print_losses, color=valid_color, linestyle="-", alpha=0.3, lw=0.5)

            plt.xlim(0, epoch+1)
            plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.savefig(os.path.join(self.result_dir + "/" + tag, "loss.png"))
            plt.close()

            # 精度のプロット
            plt.figure(figsize=(6, 4), layout="tight")
            plt.xlabel("epoch")
            plt.ylabel("accuracy")

            # 訓練データのTop-1精度
            print_x, print_accs = np.empty(0), np.empty(0)
            for epoch, accs in logs["train"]["top1_acc"].items():
                epoch = int(epoch)
                print_x = np.concatenate([print_x, np.linspace(start=epoch, stop=epoch+1, num=len(accs))])
                print_accs = np.concatenate([print_accs, accs])
            moving_avg = np.convolve(print_accs, np.ones(int(len(print_accs)/100))/(len(print_accs)/100), mode="same")
            plt.plot(print_x, moving_avg, color=train_color, linestyle="-", label="top1_acc(train)", lw=1)
            if plot_whole_data:
                plt.plot(print_x, print_accs, color=train_color, linestyle="-", alpha=0.3, lw=0.5)

            # 検証データのTop-1精度
            print_x, print_accs = np.empty(0), np.empty(0)
            for epoch, accs in logs["valid"]["top1_acc"].items():
                epoch = int(epoch)
                print_x = np.concatenate([print_x, np.linspace(start=epoch, stop=epoch+1, num=len(accs))])
                print_accs = np.concatenate([print_accs, accs])
            moving_avg = np.convolve(print_accs, np.ones(int(len(print_accs)/100))/(len(print_accs)/100), mode="same")
            plt.plot(print_x, moving_avg, color=valid_color, linestyle="-", label="top1_acc(valid)", lw=1)
            if plot_whole_data:
                plt.plot(print_x, print_accs, color=valid_color, linestyle="-", alpha=0.3, lw=0.5)

            plt.xlim(0, epoch+1)
            plt.ylim(-0.01, 1.01)
            plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.savefig(os.path.join(self.result_dir + "/" + tag, "accuracy.png"))
            plt.close()

    def plot_cross_graphs(self, targets: list = [], plot_whole_data: bool = False):
        """
        複数の実験について、検証損失と精度を1つのグラフにプロット。

        Args:
            targets (list): プロット対象とする実験の名前リスト。
            plot_whole_data (bool): 全データをプロットするかのフラグ。
        """
        if len(targets) == 0: 
            targets = self.results.keys()

        # 検証損失の比較プロット
        plt.figure(figsize=(6, 4), layout="tight")
        plt.xlabel("epoch")
        plt.ylabel("validation loss")

        epoch = 0
        for i, (tag, logs) in enumerate(self.results.items()):
            if tag not in targets:
                continue

            print_x, print_losses = np.empty(0), np.empty(0)
            for epoch, losses in logs["valid"]["loss"].items():
                epoch = int(epoch)
                print_x = np.concatenate([print_x, np.linspace(start=epoch, stop=epoch+1, num=len(losses))])
                print_losses = np.concatenate([print_losses, losses])
            moving_avg = np.convolve(print_losses, np.ones(int(len(print_losses)/100))/(len(print_losses)/100), mode="same")
            plt.plot(print_x, moving_avg, color=self.color_map(i), linestyle="-", label=tag, lw=1)
            if plot_whole_data:
                plt.plot(print_x, print_losses, color=self.color_map(i), linestyle="-", alpha=0.3, lw=0.5)

        plt.xlim(0, epoch+1)
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.savefig(os.path.join(self.result_dir, "loss.png"))
        plt.close()

        # 検証精度の比較プロット
        plt.figure(figsize=(6, 4), layout="tight")
        plt.xlabel("epoch")
        plt.ylabel("validation accuracy")

        for i, (tag, logs) in enumerate(self.results.items()):
            if tag not in targets:
                continue

            # Top-1精度
            print_x, print_accs = np.empty(0), np.empty(0)
            for epoch, accs in logs["valid"]["top1_acc"].items():
                epoch = int(epoch)
                print_x = np.concatenate([print_x, np.linspace(start=epoch, stop=epoch+1, num=len(accs))])
                print_accs = np.concatenate([print_accs, accs])
            moving_avg = np.convolve(print_accs, np.ones(int(len(print_accs)/100))/(len(print_accs)/100), mode="same")
            plt.plot(print_x, moving_avg, color=self.color_map(i), linestyle="-", label=f"top1-acc({tag})", lw=1)
            if plot_whole_data:
                plt.plot(print_x, print_accs, color=self.color_map(i), linestyle="-", alpha=0.3, lw=0.5)

        plt.xlim(0, epoch+1)
        plt.ylim(-0.01, 1.01)
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.savefig(os.path.join(self.result_dir, "accuracy.png"))
        plt.close()

    def plot_attentionmap(self, model, dataset, ckpt_path: str = None, n_feature: int = 3):
        """
        モデルのアテンションマップを描画するための関数。
        モデルタイプに応じて適切な描画関数を呼び出し、結果を保存します。

        Args:
            model: 使用するモデル（例: TimeSformer, SwinTransformer3D）。
            dataset: アテンションマップ生成に使用するデータセット。
            ckpt_path (str): モデルのチェックポイントファイルパス（オプション）。
            n_feature (int): 描画するサンプル数。
        """
        # 結果を保存するディレクトリの設定
        result_dir = os.path.join(os.path.join(self.result_dir, self.label), "attention_map")

        # デバイスを選択（GPUが利用可能であればGPUを使用）
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # チェックポイントをロード（オプション）
        if ckpt_path:
            assert os.path.exists(ckpt_path), f"チェックポイント {ckpt_path} が見つかりません。"
            model.load_state_dict(torch.load(ckpt_path, weights_only=True), strict=False)

        # モデルを評価モードに設定し、勾配計算を無効化
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)

        # データセットからランダムに n_feature サンプルを選択し、入力を準備
        rnd_idx = np.random.choice(len(dataset), size=n_feature)
        clip = dataset[rnd_idx[0]][0].unsqueeze(0)  # 最初のサンプルを取得
        for i in rnd_idx[1:]:
            clip = torch.cat([clip, dataset[i][0].unsqueeze(0)], dim=0)  # 他のサンプルを結合
        clip = clip.to(device)

        # モデルタイプに応じて適切な関数を呼び出す
        if isinstance(model, TimeSformer):
            self.plot_attentionmap_timesformer(result_dir=result_dir, model=model, dataset=clip, n_feature=n_feature)
        elif isinstance(model, SwinTransformer3D):
            self.plot_attentionmap_swintransfomer(result_dir=result_dir, model=model, dataset=clip, n_feature=n_feature)
        else:
            print(f"モデルタイプ {type(model)} はサポートされていません。Attention Map の描画をスキップします。")
            return

    def plot_attentionmap_pretrain(self, model, dataset, ckpt_path: str = None, n_feature: int = 3):
        """
        事前学習フェーズ用のアテンションマップを描画する関数。
        モデルタイプに応じて適切な描画関数を呼び出し、結果を保存します。

        Args:
            model: 使用するモデル（例: TimeSformer, SwinTransformer3D）。
            dataset: アテンションマップ生成に使用するデータセット。
            ckpt_path (str): モデルのチェックポイントファイルパス（オプション）。
            n_feature (int): 描画するサンプル数。
        """
        # 結果を保存するディレクトリの設定
        result_dir = os.path.join(os.path.join(self.result_dir, self.label), "pretrain-attention_map")

        # デバイスを選択（GPUが利用可能であればGPUを使用）
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # チェックポイントをロード（オプション）
        if ckpt_path:
            assert os.path.exists(ckpt_path), f"チェックポイント {ckpt_path} が見つかりません。"
            model.load_state_dict(torch.load(ckpt_path, weights_only=True), strict=False)

        # モデルを評価モードに設定し、勾配計算を無効化
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)

        # データセットからランダムに n_feature サンプルを選択し、入力を準備
        rnd_idx = np.random.choice(len(dataset), size=n_feature)
        clip = dataset[rnd_idx[0]][0].unsqueeze(0)  # 最初のサンプルを取得
        for i in rnd_idx[1:]:
            clip = torch.cat([clip, dataset[i][0].unsqueeze(0)], dim=0)  # 他のサンプルを結合
        clip = clip.to(device)

        # モデルタイプに応じて適切な関数を呼び出す
        if isinstance(model, TimeSformer):
            self.plot_attentionmap_timesformer(result_dir=result_dir, model=model, dataset=clip, n_feature=n_feature)
        elif isinstance(model, SwinTransformer3D):
            self.plot_attentionmap_swintransfomer(result_dir=result_dir, model=model, dataset=clip, n_feature=n_feature)
        else:
            print(f"モデルタイプ {type(model)} はサポートされていません。Attention Map の描画をスキップします。")
            return

    def plot_attentionmap_timesformer(self, result_dir: str, model, dataset, n_feature: int = 3):
        """
        TimeSformer モデル用のアテンションマップを描画する関数。

        Args:
            result_dir (str): アテンションマップを保存するディレクトリ。
            model: 使用する TimeSformer モデル。
            dataset: アテンションマップ生成に使用する入力データ。
            n_feature (int): 描画するサンプル数。
        """
        from dataset.DatasetFactory import dataset_params
        from model.ModelFactory import model_params

        # 特徴マップサイズを計算
        w_featmap = dataset_params.img_size // model_params.patch_size
        h_featmap = dataset_params.img_size // model_params.patch_size

        with torch.no_grad():
            y = model(dataset)  # モデルの推論を実行（結果は使用しない）

        # アテンションを取得し、形状を再構築
        attentions = model.getAttention()
        attentions = rearrange(attentions[:, :, 0, 1:], 
                            "(B T) N (H W) -> B T N H W", 
                            B=n_feature, 
                            T=dataset_params.n_frames, 
                            H=h_featmap, 
                            W=w_featmap)

        for i in range(n_feature):
            # 結果保存用ディレクトリを作成
            img_dir = os.path.join(result_dir, f"{i:02d}")
            os.makedirs(img_dir, exist_ok=True)

            # 入力画像を保存
            grid_clip = torchvision.utils.make_grid(dataset[i].cpu(), normalize=True, scale_each=True)
            torchvision.utils.save_image(grid_clip, os.path.join(img_dir, "img_original.png"))

            # アテンションを補間し、保存する
            bicubic_attention = torch.nn.functional.interpolate(attentions[i], 
                                                                scale_factor=(model_params.patch_size, model_params.patch_size), 
                                                                mode="bicubic")
            self._save_attention_maps(bicubic_attention, img_dir, mode="bicubic")

            nearest_attention = torch.nn.functional.interpolate(attentions[i], 
                                                                scale_factor=model_params.patch_size * model_params.patch_size, 
                                                                mode="nearest")
            self._save_attention_maps(nearest_attention, img_dir, mode="nearest")

    def plot_attentionmap_swintransfomer(self, result_dir: str, model, dataset, n_feature: int = 3):
        """
        SwinTransformer モデル用のアテンションマップを描画する関数。

        Args:
            result_dir (str): アテンションマップを保存するディレクトリ。
            model: 使用する SwinTransformer モデル。
            dataset: アテンションマップ生成に使用する入力データ。
            n_feature (int): 描画するサンプル数。
        """
        from dataset.DatasetFactory import dataset_params
        from model.ModelFactory import model_params

        # 特徴マップサイズを計算
        w_featmap = dataset_params.img_size // (model_params.patch_size[1] * model_params.patch_size[2])
        h_featmap = dataset_params.img_size // (model_params.patch_size[1] * model_params.patch_size[2])

        with torch.no_grad():
            y = model(dataset)  # モデルの推論を実行（結果は使用しない）

        # アテンションを取得し、形状を再構築
        attentions = model.getAttention()
        attentions = rearrange(attentions.mean(dim=2), 
                            "B N (H W) -> B N H W", 
                            H=h_featmap, 
                            W=w_featmap)

        for i in range(n_feature):
            # 結果保存用ディレクトリを作成
            img_dir = os.path.join(result_dir, f"{i:02d}")
            os.makedirs(img_dir, exist_ok=True)

            # 入力画像を保存
            grid_clip = torchvision.utils.make_grid(dataset[i].cpu(), normalize=True, scale_each=True)
            torchvision.utils.save_image(grid_clip, os.path.join(img_dir, "img_original.png"))

            # アテンションを補間して保存
            bicubic_attention = torch.nn.functional.interpolate(attentions[i].unsqueeze(0), 
                                                                scale_factor=model_params.patch_size[1] * model_params.patch_size[2], 
                                                                mode="bicubic")
            self._save_attention_maps(bicubic_attention, img_dir, mode="bicubic")

            nearest_attention = torch.nn.functional.interpolate(attentions[i].unsqueeze(0), 
                                                                scale_factor=model_params.patch_size[1] * model_params.patch_size[2], 
                                                                mode="nearest")
            self._save_attention_maps(nearest_attention, img_dir, mode="nearest")

    def _save_attention_maps(self, attention, img_dir, mode: str):
        """
        アテンションマップを保存するヘルパー関数。

        Args:
            attention: アテンションマップテンソル。
            img_dir (str): 保存ディレクトリ。
            mode (str): 補間モードの名前（例："bicubic", "nearest"）。
        """
        mode_dir = os.path.join(img_dir, f"interpolate_{mode}")
        os.makedirs(mode_dir, exist_ok=True)

        heatmap_img_dir = os.path.join(mode_dir, "mean_attn_heatmap_for_frames")
        os.makedirs(heatmap_img_dir, exist_ok=True)
        for j in range(attention.size(0)):
            fname = os.path.join(heatmap_img_dir, f"attnheatmap_frame-{j:02d}.png")
            plt.imsave(fname=fname, arr=attention[j].mean(dim=0).cpu().numpy(), cmap='viridis')

        head_img_dir = os.path.join(mode_dir, "attentions_each_heads")
        os.makedirs(head_img_dir, exist_ok=True)
        for j in range(attention.size(1)):
            head_attention = torchvision.utils.make_grid(attention[:, j].unsqueeze(1), normalize=True, scale_each=True)
            torchvision.utils.save_image(head_attention, os.path.join(head_img_dir, f"attention_head-{j:02d}.png"))

# メイン部分：指定ディレクトリのデータを解析
if __name__ == "__main__":
    analyzer = Analyzer("./results")
    analyzer.plot_graphs()
    analyzer.plot_cross_graphs()