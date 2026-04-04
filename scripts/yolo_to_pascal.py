"""
yolo_to_pascal.py  –  YOLO txt → Pascal VOC XML 変換スクリプト

"""

import argparse
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# 対応画像拡張子（探索順）
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
def load_class_list(class_file: Path) -> list[str]:
    if class_file.exists():
        classes = [l.strip() for l in class_file.read_text().splitlines() if l.strip()]
        logger.info(f"クラスリスト読み込み: {len(classes)} クラス ({class_file})")
        return classes
    logger.warning(f"classes.txt が見つかりません: {class_file}")
    return []


def find_image(stem: str, img_dir: Path) -> Path | None:
    """ステム名に対応する画像ファイルをどの拡張子でも探す。"""
    for ext in IMAGE_EXTENSIONS:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None


def get_image_info(img_path: Path) -> tuple[int, int, int] | None:
    """(width, height, depth) を返す。PIL がなければ None。"""
    if not PIL_AVAILABLE:
        logger.error("PIL がインストールされていません。pip install Pillow を実行してください。")
        return None
    with Image.open(img_path) as img:
        w, h = img.size
        mode_to_depth = {"RGB": 3, "RGBA": 4, "L": 1, "1": 1}
        depth = mode_to_depth.get(img.mode, 3)
        return w, h, depth


def clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def indent_xml(elem, level: int = 0) -> None:
    """ET の要素に再帰的にインデントを追加する（Python 3.9+ の ET.indent の代替）。"""
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad
    if not level:
        elem.tail = "\n"


# ──────────────────────────────────────────────
# 変換コア
# ──────────────────────────────────────────────
def yolo_to_voc_single(
    label_path: Path,
    img_dir: Path,
    voc_save_dir: Path,
    class_list: list[str],
    dry_run: bool,
) -> dict:
    """
    1 件の YOLO txt を VOC XML に変換する。

    Returns:
        dict: {"ok": bool, "skipped_objs": int, "msg": str}
    """
    result = {"ok": False, "skipped_objs": 0, "msg": ""}

    img_path = find_image(label_path.stem, img_dir)
    if img_path is None:
        result["msg"] = f"対応画像が見つかりません: {label_path.stem}.*"
        return result

    info = get_image_info(img_path)
    if info is None:
        result["msg"] = f"画像情報取得失敗: {img_path}"
        return result

    img_width, img_height, depth = info

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = img_path.name
    ET.SubElement(annotation, "path").text = str(img_path)
    size_el = ET.SubElement(annotation, "size")
    ET.SubElement(size_el, "width").text  = str(img_width)
    ET.SubElement(size_el, "height").text = str(img_height)
    ET.SubElement(size_el, "depth").text  = str(depth)

    lines = label_path.read_text().splitlines()
    valid_obj_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            logger.warning(f"  フォーマット不正: '{line}' in {label_path.name}")
            result["skipped_objs"] += 1
            continue

        try:
            cls_id   = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            bw       = float(parts[3])
            bh       = float(parts[4])
        except ValueError:
            logger.warning(f"  数値変換失敗: '{line}' in {label_path.name}")
            result["skipped_objs"] += 1
            continue

        if cls_id < 0 or cls_id >= len(class_list):
            logger.warning(f"  クラスID範囲外: {cls_id} (クラス数={len(class_list)}) in {label_path.name}")
            result["skipped_objs"] += 1
            continue

        # 正規化座標 → ピクセル座標（クランプ付き）
        xmin = clamp_int(int((x_center - bw / 2) * img_width),  0, img_width  - 1)
        ymin = clamp_int(int((y_center - bh / 2) * img_height), 0, img_height - 1)
        xmax = clamp_int(int((x_center + bw / 2) * img_width),  1, img_width)
        ymax = clamp_int(int((y_center + bh / 2) * img_height), 1, img_height)

        if xmax <= xmin or ymax <= ymin:
            logger.warning(f"  無効な BBox (クランプ後): cls={cls_id} in {label_path.name}")
            result["skipped_objs"] += 1
            continue

        obj_el = ET.SubElement(annotation, "object")
        ET.SubElement(obj_el, "name").text = class_list[cls_id]
        ET.SubElement(obj_el, "pose").text = "Unspecified"
        ET.SubElement(obj_el, "truncated").text = "0"
        ET.SubElement(obj_el, "difficult").text = "0"
        bndbox_el = ET.SubElement(obj_el, "bndbox")
        ET.SubElement(bndbox_el, "xmin").text = str(xmin)
        ET.SubElement(bndbox_el, "ymin").text = str(ymin)
        ET.SubElement(bndbox_el, "xmax").text = str(xmax)
        ET.SubElement(bndbox_el, "ymax").text = str(ymax)
        valid_obj_count += 1

    if valid_obj_count == 0:
        result["msg"] = f"有効なオブジェクトなし: {label_path.name}"
        return result

    # インデント付きで書き出し
    indent_xml(annotation)
    tree = ET.ElementTree(annotation)

    out_path = voc_save_dir / (label_path.stem + ".xml")
    if not dry_run:
        voc_save_dir.mkdir(parents=True, exist_ok=True)
        tree.write(out_path, encoding="utf-8", xml_declaration=True)

    result["ok"]  = True
    result["msg"] = str(out_path)
    return result


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="YOLO txt → Pascal VOC XML 変換ツール",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--yolo_labels_dir", type=str, required=True,  help="YOLO txt ファイルのディレクトリ")
    parser.add_argument("--yolo_images_dir", type=str, required=True,  help="対応する画像ファイルのディレクトリ")
    parser.add_argument("--voc_save_dir",    type=str, required=True,  help="VOC XML の出力先ディレクトリ")
    parser.add_argument("--class_file",      type=str, default="classes.txt", help="classes.txt のパス")
    parser.add_argument("--error_log",       type=str, default="convert_errors.log", help="エラーログの出力先")
    parser.add_argument("--dry_run",         action="store_true", help="実際には書き込まずに動作確認")
    args = parser.parse_args()

    label_dir   = Path(args.yolo_labels_dir)
    img_dir     = Path(args.yolo_images_dir)
    voc_save_dir= Path(args.voc_save_dir)
    class_file  = Path(args.class_file)

    if args.dry_run:
        logger.info("=" * 40)
        logger.info("DRY-RUN モード: ファイルは書き込まれません")
        logger.info("=" * 40)

    class_list  = load_class_list(class_file)
    label_files = sorted(label_dir.glob("*.txt"))

    if not label_files:
        logger.error(f"txt ファイルが見つかりません: {label_dir}")
        return

    logger.info(f"{len(label_files)} 件の txt を変換します")

    ok_count      = 0
    skip_count    = 0
    skip_obj_total= 0
    error_log     = []

    iterator = tqdm(label_files, desc="変換中") if TQDM_AVAILABLE else label_files
    for label_path in iterator:
        # classes.txt 自体はスキップ
        if label_path.name == "classes.txt":
            continue

        res = yolo_to_voc_single(label_path, img_dir, voc_save_dir, class_list, args.dry_run)
        if res["ok"]:
            ok_count += 1
        else:
            skip_count += 1
            error_log.append(res["msg"])
            logger.warning(f"スキップ: {res['msg']}")

        skip_obj_total += res["skipped_objs"]

    if error_log and not args.dry_run:
        Path(args.error_log).write_text("\n".join(error_log) + "\n")
        logger.info(f"エラーログを保存: {args.error_log}")

    # サマリ表示
    print("\n" + "=" * 45)
    print(f"  変換完了サマリ")
    print("=" * 45)
    print(f"  成功:             {ok_count} / {len(label_files)} 件")
    print(f"  スキップ(ファイル):   {skip_count} 件")
    print(f"  スキップ(オブジェクト): {skip_obj_total} 件")
    if args.dry_run:
        print("  ※ DRY-RUN: ファイルは書き込まれていません")
    print("=" * 45)


if __name__ == "__main__":
    main()
