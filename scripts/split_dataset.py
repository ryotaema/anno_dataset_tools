#train,val,testの自動分割

import glob 
import os
import random

dataset = "YOLOtest"#データセット元のディレクトリ名(YOLOtest)の指定

#labelディレクトリ内の.txtを取得
img_list = glob.glob(os.path.join(dataset + "/label", "*.txt"))

#シャッフル
random.shuffle(img_list)

#データの分割
num_data = len(img_list)
num_train = int(num_data * 0.6)
num_val = int(num_data * 0.2)
num_test = num_data - num_train - num_val
"""
ここのパラメータを変更することでデータセットないの分割比率を変更出来ます
num_train = int(num_data * 0.6)←の0.6と　num_val = int(num_data * 0.2)←の0.2を好きに変更してください。
この場合だと(学習、検証、テスト)=(6:2:2)の比率です。
"""

#辞書に隔離
split_dict = {
    "train": img_list[:num_train],
    "valid": img_list[num_train:num_train + num_val],
    "test": img_list[num_train + num_val:]
}

# ディレクトリを作成
for folder in ["images", "labels"]:
    # images/train, images/valid, images/test のように作成
    for split in ["train", "valid", "test"]:
        dir_path = os.path.join(dataset, folder, split)
        os.makedirs(dir_path, exist_ok=True)
        
# ファイルをコピー
for split, paths in split_dict.items():
    for txt_path in paths:
        # 対応する画像ファイルのパスを取得
        img_path = txt_path.replace("label", "image").replace(".txt", ".jpg")

        # コピー先のディレクトリ
        label_dest = os.path.join(dataset, "labels", split)
        img_dest = os.path.join(dataset, "images", split)

        # ファイルをコピー
        os.system(f"cp {txt_path} {label_dest}")
        os.system(f"cp {img_path} {img_dest}")
        
"""
ChatGPT様より
改善ポイント
クロスプラットフォーム対応

os.system() の cp コマンドはLinux/Macでしか動作しません。
Windows環境でも動作するようにするには、次のように shutil を使うのが望ましいです。
python
コードをコピーする
import shutil
shutil.copy(txt_path, label_dest)
shutil.copy(img_path, img_dest)
エラーハンドリング

もし画像ファイルが存在しない場合に備え、try-except で例外処理を入れると、さらに堅牢なコードになります。
"""
        
