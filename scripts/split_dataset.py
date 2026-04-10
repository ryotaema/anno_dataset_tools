"""
split_dataset.py  –  YOLO形式データセットの train / val / test 分割スクリプト
（GUIフォルダ選択対応版）
"""

import argparse
import logging
import random
import shutil
import tkinter as tk
from tkinter import filedialog
from collections import Counter, defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# 対応画像拡張子
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]


# ──────────────────────────────────────────────
# ロギング設定
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────
def find_image(stem: str, img_dir: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None


def get_primary_class(label_path: Path) -> str | None:
    """
    ラベルファイルから最初のクラスIDを取得する（stratified split 用）。
    読めない・空の場合は None を返す。
    """
    try:
        lines = label_path.read_text().splitlines()
        for line in lines:
            parts = line.strip().split()
            if parts:
                return parts[0]  # 最初のクラスID（文字列）を返す
    except Exception:
        pass
    return None


def split_indices_stratified(
    label_files: list[Path],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[Path]]:
    """
    クラスバランスを維持した stratified split。
    各クラスで train/val/test 比率が同等になるよう分割する。
    """
    rng = random.Random(seed)
    train_r, val_r, test_r = ratios

    # クラスごとにグループ化
    class_groups: dict[str, list[Path]] = defaultdict(list)
    no_label = []
    for lf in label_files:
        cls = get_primary_class(lf)
        if cls is None:
            no_label.append(lf)
        else:
            class_groups[cls].append(lf)

    train_files, val_files, test_files = [], [], []

    for cls, files in class_groups.items():
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, round(n * train_r))
        n_val   = max(0, round(n * val_r))
        # 残りは全て test（端数対応）
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])

    # ラベルなしは train に追加
    if no_label:
        logger.warning(f"ラベルなし/空ファイル: {len(no_label)} 件 → trainに割り当て")
        train_files.extend(no_label)

    splits = {"train": train_files, "val": val_files}
    if test_r > 0:
        splits["test"] = test_files
    return splits


def split_indices_random(
    label_files: list[Path],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[Path]]:
    """シンプルなランダム分割。"""
    rng = random.Random(seed)
    files = list(label_files)
    rng.shuffle(files)

    n = len(files)
    train_r, val_r, test_r = ratios
    n_train = round(n * train_r)
    n_val   = round(n * val_r)

    splits = {
        "train": files[:n_train],
        "val":   files[n_train:n_train + n_val],
    }
    if test_r > 0:
        splits["test"] = files[n_train + n_val:]
    return splits


def transfer_file(src: Path, dst: Path, move: bool, dry_run: bool) -> bool:
    """ファイルをコピー or 移動する。"""
    if dry_run:
        return True
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(src), dst)
        else:
            shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.error(f"転送エラー: {src} → {dst}: {e}")
        return False


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="YOLO 形式データセットの train/val/test 分割ツール",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # required=True を外し、デフォルト値を None に変更
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="データセットのルートディレクトリ（指定しない場合はダイアログで選択）")
    parser.add_argument("--train_ratio", type=float, default=0.8,  help="train の割合")
    parser.add_argument("--val_ratio",   type=float, default=0.2,  help="val の割合")
    parser.add_argument("--test_ratio",  type=float, default=0.0,  help="test の割合（0で test なし）")
    parser.add_argument("--seed",        type=int,   default=42,   help="乱数シード（再現性確保）")
    parser.add_argument("--stratified",  action="store_true",      help="クラスバランスを維持した stratified split を使用")
    parser.add_argument("--move",        action="store_true",      help="コピーの代わりに移動（元ファイルを削除）")
    parser.add_argument("--dry_run",     action="store_true",      help="実際には書き込まずに動作確認")
    args = parser.parse_args()

    dataset_dir_str = args.dataset_dir

    # コマンドライン引数でディレクトリが指定されていない場合、ダイアログを開く
    if dataset_dir_str is None:
        root = tk.Tk()
        root.withdraw()  # 余分なメインウィンドウを隠す
        dataset_dir_str = filedialog.askdirectory(
            title="データセットのルートフォルダを選択してください (images/ と labels/ を含むフォルダ)"
        )
        
        # キャンセルされた場合は終了
        if not dataset_dir_str:
            logger.error("フォルダ選択がキャンセルされました。スクリプトを終了します。")
            return

    dataset_dir = Path(dataset_dir_str).expanduser().resolve()
    labels_dir  = dataset_dir / "labels"
    images_dir  = dataset_dir / "images"

    if not labels_dir.exists():
        logger.error(f"labels ディレクトリが見つかりません: {labels_dir}")
        return
    if not images_dir.exists():
        logger.error(f"images ディレクトリが見つかりません: {images_dir}")
        return

    # 比率の検証
    train_r, val_r, test_r = args.train_ratio, args.val_ratio, args.test_ratio
    total = train_r + val_r + test_r
    if not (0.99 <= total <= 1.01):
        logger.error(f"比率の合計が 1.0 になりません: {train_r}+{val_r}+{test_r}={total:.3f}")
        return

    if args.dry_run:
        logger.info("=" * 40)
        logger.info("DRY-RUN モード: ファイルは書き込まれません")
        logger.info("=" * 40)

    # labels/ 直下の .txt のみ取得（サブフォルダ・classes.txt 除外）
    label_files = [
        f for f in labels_dir.glob("*.txt")
        if f.parent == labels_dir and f.name != "classes.txt"
    ]

    if not label_files:
        logger.error(f"有効な .txt ファイルが見つかりません: {labels_dir}")
        return

    logger.info(f"対象データセット: {dataset_dir}")
    logger.info(f"総データ数: {len(label_files)} 件")
    logger.info(f"分割比率  : train={train_r:.0%} / val={val_r:.0%} / test={test_r:.0%}")
    logger.info(f"Seed      : {args.seed}  / Stratified: {args.stratified}")
    logger.info(f"転送方法  : {'move' if args.move else 'copy'}")

    # 分割実行
    ratios = (train_r, val_r, test_r)
    if args.stratified:
        splits = split_indices_stratified(label_files, ratios, args.seed)
    else:
        splits = split_indices_random(label_files, ratios, args.seed)

    # ファイル転送
    stats: dict[str, Counter] = {s: Counter(ok=0, missing=0, error=0) for s in splits}

    for split_name, files in splits.items():
        logger.info(f"--- {split_name}: {len(files)} 件 ---")
        iterator = tqdm(files, desc=split_name) if TQDM_AVAILABLE else files

        for label_src in iterator:
            img_src = find_image(label_src.stem, images_dir)
            if img_src is None:
                logger.warning(f"  画像が見つかりません: {label_src.stem}.*")
                stats[split_name]["missing"] += 1
                continue

            label_dst = dataset_dir / "labels" / split_name / label_src.name
            img_dst   = dataset_dir / "images" / split_name / img_src.name

            ok_label = transfer_file(label_src, label_dst, args.move, args.dry_run)
            ok_img   = transfer_file(img_src,   img_dst,   args.move, args.dry_run)

            if ok_label and ok_img:
                stats[split_name]["ok"] += 1
            else:
                stats[split_name]["error"] += 1

    # サマリ表示
    print("\n" + "=" * 50)
    print("  分割完了サマリ")
    print("=" * 50)
    total_ok = 0
    for split_name, c in stats.items():
        total_ok += c["ok"]
        line = f"  {split_name:<8}: {c['ok']:>5} 件転送"
        if c["missing"]:
            line += f"  / 画像なし: {c['missing']} 件"
        if c["error"]:
            line += f"  / エラー: {c['error']} 件"
        print(line)
    print(f"\n  合計転送: {total_ok} 件")
    if args.dry_run:
        print("  ※ DRY-RUN: ファイルは書き込まれていません")
    print("=" * 50)


if __name__ == "__main__":
    main()