"""
pascal_to_yolo.py  –  Pascal VOC XML → YOLO txt 変換スクリプト

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
    """classes.txt を読み込む。存在しない場合は空リストを返す。"""
    if class_file.exists():
        classes = [l.strip() for l in class_file.read_text().splitlines() if l.strip()]
        logger.info(f"クラスリスト読み込み: {len(classes)} クラス ({class_file})")
        return classes
    logger.warning(f"classes.txt が見つかりません: {class_file}  → 空リストで開始します")
    return []


def save_class_list(class_file: Path, class_list: list[str], dry_run: bool = False) -> None:
    """classes.txt を保存する。"""
    if dry_run:
        logger.info(f"[DRY-RUN] classes.txt を保存しません: {class_file}")
        return
    class_file.parent.mkdir(parents=True, exist_ok=True)
    class_file.write_text("\n".join(class_list) + "\n")
    logger.info(f"classes.txt を更新: {class_file}  ({len(class_list)} クラス)")


def get_image_size(xml_root, img_dir: Path, filename: str) -> tuple[int, int] | None:
    """
    画像サイズを取得する。
    優先順位: 1) XML の <size> タグ  2) 実画像ファイルを開いて取得
    """
    size_el = xml_root.find("size")
    if size_el is not None:
        w_el = size_el.find("width")
        h_el = size_el.find("height")
        if w_el is not None and h_el is not None:
            w, h = int(w_el.text), int(h_el.text)
            if w > 0 and h > 0:
                return w, h

    # XML に size がない場合は実ファイルから取得
    if not PIL_AVAILABLE:
        logger.error("PIL がインストールされていないため画像サイズを取得できません。")
        return None

    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"):
        img_path = img_dir / (Path(filename).stem + ext)
        if img_path.exists():
            with Image.open(img_path) as img:
                return img.size  # (width, height)

    return None


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """値を [lo, hi] に収める。"""
    return max(lo, min(hi, value))


# ──────────────────────────────────────────────
# 変換コア
# ──────────────────────────────────────────────
def voc_to_yolo_single(
    xml_path: Path,
    img_dir: Path,
    yolo_save_dir: Path,
    class_list: list[str],
    dry_run: bool,
) -> dict:
    """
    1 件の XML を YOLO txt に変換する。

    Returns:
        dict: {"ok": bool, "new_classes": list, "skipped_objs": int, "msg": str}
    """
    result = {"ok": False, "new_classes": [], "skipped_objs": 0, "msg": ""}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        result["msg"] = f"XML 解析エラー: {xml_path} – {e}"
        return result

    filename_el = root.find("filename")
    filename = filename_el.text.strip() if filename_el is not None else xml_path.stem + ".jpg"

    size = get_image_size(root, img_dir, filename)
    if size is None:
        result["msg"] = f"画像サイズ取得失敗: {xml_path}"
        return result

    img_width, img_height = size
    yolo_lines = []

    for obj in root.findall("object"):
        cls_name_el = obj.find("name")
        if cls_name_el is None:
            result["skipped_objs"] += 1
            continue
        cls_name = cls_name_el.text.strip()

        # 新クラスは自動追加（スキップしない）
        if cls_name not in class_list:
            logger.info(f"  新クラスを追加: '{cls_name}'")
            class_list.append(cls_name)
            result["new_classes"].append(cls_name)

        cls_id = class_list.index(cls_name)

        bndbox = obj.find("bndbox")
        if bndbox is None:
            result["skipped_objs"] += 1
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except (TypeError, ValueError):
            result["skipped_objs"] += 1
            continue

        if xmax <= xmin or ymax <= ymin:
            logger.warning(f"  無効な BBox (xmin={xmin},ymin={ymin},xmax={xmax},ymax={ymax}): {xml_path.name}")
            result["skipped_objs"] += 1
            continue

        x_center = clamp(((xmin + xmax) / 2) / img_width)
        y_center = clamp(((ymin + ymax) / 2) / img_height)
        box_w    = clamp((xmax - xmin) / img_width)
        box_h    = clamp((ymax - ymin) / img_height)

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    if not yolo_lines:
        result["msg"] = f"有効なオブジェクトなし: {xml_path.name}"
        return result

    out_path = yolo_save_dir / (xml_path.stem + ".txt")
    if not dry_run:
        yolo_save_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(yolo_lines) + "\n")

    result["ok"] = True
    result["msg"] = str(out_path)
    return result


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pascal VOC XML → YOLO txt 変換ツール",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--voc_dir",      type=str, required=True,  help="VOC XML ファイルのディレクトリ")
    parser.add_argument("--img_dir",      type=str, required=True,  help="画像ファイルのディレクトリ（size タグがない場合に使用）")
    parser.add_argument("--yolo_save_dir",type=str, required=True,  help="YOLO txt の出力先ディレクトリ")
    parser.add_argument("--class_file",   type=str, default="classes.txt", help="classes.txt のパス")
    parser.add_argument("--error_log",    type=str, default="convert_errors.log", help="エラーログの出力先")
    parser.add_argument("--dry_run",      action="store_true", help="実際には書き込まずに動作確認")
    args = parser.parse_args()

    voc_dir       = Path(args.voc_dir)
    img_dir       = Path(args.img_dir)
    yolo_save_dir = Path(args.yolo_save_dir)
    class_file    = Path(args.class_file)

    if args.dry_run:
        logger.info("=" * 40)
        logger.info("DRY-RUN モード: ファイルは書き込まれません")
        logger.info("=" * 40)

    class_list = load_class_list(class_file)
    xml_files  = sorted(voc_dir.glob("*.xml"))

    if not xml_files:
        logger.error(f"XML ファイルが見つかりません: {voc_dir}")
        return

    logger.info(f"{len(xml_files)} 件の XML を変換します")

    # 統計カウンタ
    ok_count      = 0
    skip_count    = 0
    skip_obj_total= 0
    all_new_cls   = []
    error_log     = []

    iterator = tqdm(xml_files, desc="変換中") if TQDM_AVAILABLE else xml_files
    for xml_path in iterator:
        res = voc_to_yolo_single(xml_path, img_dir, yolo_save_dir, class_list, args.dry_run)
        if res["ok"]:
            ok_count += 1
        else:
            skip_count += 1
            error_log.append(res["msg"])
            logger.warning(f"スキップ: {res['msg']}")

        skip_obj_total += res["skipped_objs"]
        all_new_cls.extend(res["new_classes"])

    # classes.txt 更新
    if all_new_cls:
        save_class_list(class_file, class_list, dry_run=args.dry_run)

    # エラーログ保存
    if error_log and not args.dry_run:
        Path(args.error_log).write_text("\n".join(error_log) + "\n")
        logger.info(f"エラーログを保存: {args.error_log}")

    # サマリ表示
    print("\n" + "=" * 45)
    print(f"  変換完了サマリ")
    print("=" * 45)
    print(f"  成功:           {ok_count} / {len(xml_files)} 件")
    print(f"  スキップ(ファイル): {skip_count} 件")
    print(f"  スキップ(オブジェクト): {skip_obj_total} 件")
    if all_new_cls:
        print(f"  新クラス追加:   {list(set(all_new_cls))}")
    if args.dry_run:
        print("  ※ DRY-RUN: ファイルは書き込まれていません")
    print("=" * 45)


if __name__ == "__main__":
    main()
