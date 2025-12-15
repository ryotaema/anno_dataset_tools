import glob 
import os
import random
import shutil

# --- 設定 ---
dataset_path = "~/pepper_ws/detaset_storage/avoa_data"
dataset = os.path.expanduser(dataset_path)

# 画像とラベルの親ディレクトリ
labels_dir = os.path.join(dataset, "labels")
images_dir = os.path.join(dataset, "images")

# 対応する画像拡張子のリスト（ここに追加すれば他の形式も対応可能）
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG", ".JPEG"]

# --- 1. ファイルリストの取得 ---
# labels フォルダ直下の .txt ファイルのみを取得
txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))

# フィルタリング: 
# 1. サブフォルダ（train/valなど）内のファイルは除外
# 2. classes.txt は除外
txt_files = [
    f for f in txt_files 
    if os.path.dirname(f) == labels_dir 
    and os.path.basename(f) != "classes.txt"
]

if not txt_files:
    print(f"エラー: {labels_dir} に有効な .txt ファイルが見つかりません。")
    exit()

print(f"総データ数 (classes.txt除く): {len(txt_files)}")

# --- 2. シャッフルと分割 (6:4) ---
random.shuffle(txt_files)

num_data = len(txt_files)
num_train = int(num_data * 0.6)
# 残りを val に
num_val = num_data - num_train 

print(f"Train: {num_train}, Val: {num_val} に分割します。")

split_dict = {
    "train": txt_files[:num_train],
    "val": txt_files[num_train:]
}

# --- 3. ディレクトリ作成 ---
for folder in ["images", "labels"]:
    for split in ["train", "val"]:
        dir_path = os.path.join(dataset, folder, split)
        os.makedirs(dir_path, exist_ok=True)

# --- 4. 画像を探してコピー ---
copy_count = 0
missing_count = 0

print("処理を開始します...")

for split, paths in split_dict.items():
    for txt_path in paths:
        # ファイル名（拡張子なし）を取得 (例: /.../abc.txt -> abc)
        basename = os.path.splitext(os.path.basename(txt_path))[0]
        
        # 対応する画像を探す
        found_img_path = None
        for ext in valid_extensions:
            potential_path = os.path.join(images_dir, basename + ext)
            if os.path.exists(potential_path):
                found_img_path = potential_path
                break # 見つかったらループを抜ける
        
        # 画像が見つからなかった場合
        if found_img_path is None:
            print(f"警告: 画像が見つかりません (ID: {basename})")
            missing_count += 1
            continue

        # コピー先のパス決定
        # 画像の拡張子は元のファイルに合わせる (found_img_pathから取得)
        img_filename = os.path.basename(found_img_path)
        
        label_dest = os.path.join(dataset, "labels", split, os.path.basename(txt_path))
        img_dest = os.path.join(dataset, "images", split, img_filename)

        # ファイルをコピー
        try:
            shutil.copy(txt_path, label_dest)
            shutil.copy(found_img_path, img_dest)
            copy_count += 1
        except Exception as e:
            print(f"コピーエラー: {e}")

print("-" * 30)
print(f"処理完了: {copy_count} 組コピーしました。")

if missing_count > 0:
    print(f"警告: {missing_count} 個のラベルに対応する画像が見つかりませんでした。")