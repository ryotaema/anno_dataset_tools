"""
annotation_validator.py  –  YOLO形式アノテーション品質チェックスクリプト

検出できる問題:
  [ファイル単位]
    ERROR  - フォーマット不正（列数不足・数値変換失敗）
    ERROR  - クラスID が classes.txt の範囲外
    ERROR  - 座標が [0, 1] の範囲外（負値・1超え）
    WARNING - BBox の幅/高さがゼロ以下
    WARNING - BBox が極小（--min_size 未満）
    WARNING - BBox が極大（--max_size 超え）

  [データセット全体]
    ERROR  - 同一画像内で IoU > --iou_thresh の重複ラベルペア
    WARNING - ラベルに対応する画像が存在しない（孤立ラベル）
    WARNING - 画像に対応するラベルが存在しない（孤立画像）
    WARNING - クラスごとのサンプル数が --min_samples 未満
    INFO   - 空ラベルファイル（background画像として意図的な場合もある）

使い方:
  python annotation_validator.py \\
      --labels_dir ./dataset/labels \\
      --images_dir ./dataset/images \\
      --class_file ./classes.txt \\
      --iou_thresh 0.95 \\
      --min_size 0.005 \\
      --max_size 0.99 \\
      --min_samples 10 \\
      --report_json ./validation_report.json

  CI/CD 連携:
    exit code 0 → ERROR なし（学習続行可）
    exit code 1 → ERROR あり（学習をブロック）
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# データ構造
# ──────────────────────────────────────────────
@dataclass
class BBox:
    cls_id: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.cx - self.w / 2

    @property
    def y1(self) -> float:
        return self.cy - self.h / 2

    @property
    def x2(self) -> float:
        return self.cx + self.w / 2

    @property
    def y2(self) -> float:
        return self.cy + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class Issue:
    level: str      # "ERROR" | "WARNING" | "INFO"
    code: str       # 機械可読なエラーコード
    file: str       # 対象ファイル名
    line: int       # 行番号（-1 = ファイル全体）
    message: str    # 人間が読むメッセージ


@dataclass
class ValidationResult:
    total_files: int = 0
    total_objects: int = 0
    empty_files: int = 0
    issues: list = field(default_factory=list)
    class_counts: dict = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "WARNING")

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "INFO")


# ──────────────────────────────────────────────
# IoU 計算
# ──────────────────────────────────────────────
def compute_iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    union_area = a.area + b.area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


# ──────────────────────────────────────────────
# ファイル単位チェック
# ──────────────────────────────────────────────
def validate_file(
    label_path: Path,
    class_list: list[str],
    min_size: float,
    max_size: float,
    result: ValidationResult,
) -> list[BBox]:
    """
    1 件の YOLO txt を検証し、有効な BBox リストを返す。
    問題は result.issues に追加する。
    """
    result.total_files += 1
    bboxes: list[BBox] = []
    fname = label_path.name

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        result.issues.append(Issue("ERROR", "FILE_READ_ERROR", fname, -1, f"読み込みエラー: {e}"))
        return bboxes

    # 空ファイル（background 画像として意図的な場合もある）
    non_empty_lines = [l for l in lines if l.strip()]
    if not non_empty_lines:
        result.empty_files += 1
        result.issues.append(Issue("INFO", "EMPTY_LABEL", fname, -1, "ラベルが空（background画像の可能性）"))
        return bboxes

    num_classes = len(class_list)

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue

        parts = line.split()

        # 列数チェック
        if len(parts) < 5:
            result.issues.append(Issue(
                "ERROR", "FORMAT_INVALID", fname, line_no,
                f"列数不足 ({len(parts)}列): '{line}'"
            ))
            continue

        # 型変換チェック
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            result.issues.append(Issue(
                "ERROR", "FORMAT_INVALID", fname, line_no,
                f"数値変換失敗: '{line}'"
            ))
            continue

        # クラスID チェック
        if num_classes > 0 and (cls_id < 0 or cls_id >= num_classes):
            result.issues.append(Issue(
                "ERROR", "CLASS_OUT_OF_RANGE", fname, line_no,
                f"クラスID {cls_id} が範囲外 (0〜{num_classes - 1}): classes.txt を確認してください"
            ))
            # クラスIDが不正でも座標チェックは続行

        # 座標範囲チェック
        coord_ok = True
        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            if val < 0.0 or val > 1.0:
                result.issues.append(Issue(
                    "ERROR", "COORD_OUT_OF_RANGE", fname, line_no,
                    f"{name}={val:.6f} が [0, 1] 範囲外"
                ))
                coord_ok = False

        # BBox 幅/高さのゼロ以下チェック
        if w <= 0 or h <= 0:
            result.issues.append(Issue(
                "ERROR", "BBOX_ZERO_SIZE", fname, line_no,
                f"BBox の幅または高さがゼロ以下 (w={w:.6f}, h={h:.6f})"
            ))
            coord_ok = False

        if not coord_ok:
            continue

        # サイズ異常チェック（警告のみ）
        area = w * h
        if area < min_size ** 2:
            result.issues.append(Issue(
                "WARNING", "BBOX_TOO_SMALL", fname, line_no,
                f"BBox が極小 (w={w:.4f}, h={h:.4f}, area={area:.6f} < {min_size**2:.6f})"
            ))
        if w > max_size or h > max_size:
            result.issues.append(Issue(
                "WARNING", "BBOX_TOO_LARGE", fname, line_no,
                f"BBox が極大 (w={w:.4f}, h={h:.4f})"
            ))

        bbox = BBox(cls_id, cx, cy, w, h)
        bboxes.append(bbox)
        result.total_objects += 1

        # クラスカウント
        cls_key = class_list[cls_id] if (0 <= cls_id < num_classes) else f"unknown_{cls_id}"
        result.class_counts[cls_key] = result.class_counts.get(cls_key, 0) + 1

    return bboxes


# ──────────────────────────────────────────────
# データセット全体チェック
# ──────────────────────────────────────────────
def check_duplicate_labels(
    fname: str,
    bboxes: list[BBox],
    iou_thresh: float,
    result: ValidationResult,
) -> None:
    """同一画像内で IoU > iou_thresh の重複ラベルペアを検出する。"""
    n = len(bboxes)
    for i in range(n):
        for j in range(i + 1, n):
            iou = compute_iou(bboxes[i], bboxes[j])
            if iou > iou_thresh:
                result.issues.append(Issue(
                    "ERROR", "DUPLICATE_LABEL", fname, -1,
                    f"BBox[{i}](cls={bboxes[i].cls_id}) と BBox[{j}](cls={bboxes[j].cls_id}) の IoU={iou:.3f} > {iou_thresh}"
                ))


def check_orphan_files(
    labels_dir: Path,
    images_dir: Path,
    result: ValidationResult,
) -> None:
    """ラベルなし画像・画像なしラベルを検出する。"""
    label_stems = {p.stem for p in labels_dir.glob("*.txt") if p.name != "classes.txt"}
    image_stems = {p.stem for p in images_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS}

    for stem in sorted(label_stems - image_stems):
        result.issues.append(Issue(
            "WARNING", "ORPHAN_LABEL", stem + ".txt", -1,
            f"対応する画像ファイルが見つかりません"
        ))

    for stem in sorted(image_stems - label_stems):
        result.issues.append(Issue(
            "WARNING", "ORPHAN_IMAGE", stem + ".*", -1,
            f"対応するラベルファイルが見つかりません"
        ))


def check_class_imbalance(
    class_list: list[str],
    class_counts: dict[str, int],
    min_samples: int,
    result: ValidationResult,
) -> None:
    """クラスごとのサンプル数が min_samples 未満のものを警告する。"""
    if not class_list:
        return
    for cls_name in class_list:
        count = class_counts.get(cls_name, 0)
        if count < min_samples:
            result.issues.append(Issue(
                "WARNING", "CLASS_IMBALANCE", "dataset", -1,
                f"クラス '{cls_name}' のサンプル数が少ない ({count} < {min_samples})"
            ))


# ──────────────────────────────────────────────
# レポート出力
# ──────────────────────────────────────────────
LEVEL_COLOR = {
    "ERROR":   "\033[91m",   # 赤
    "WARNING": "\033[93m",   # 黄
    "INFO":    "\033[96m",   # シアン
    "RESET":   "\033[0m",
}


def print_report(result: ValidationResult, class_list: list[str], use_color: bool = True) -> None:
    def color(level: str, text: str) -> str:
        if not use_color:
            return text
        return LEVEL_COLOR.get(level, "") + text + LEVEL_COLOR["RESET"]

    print("\n" + "=" * 60)
    print("  アノテーション検証レポート")
    print("=" * 60)
    print(f"  検査ファイル数   : {result.total_files}")
    print(f"  総オブジェクト数 : {result.total_objects}")
    print(f"  空ラベルファイル : {result.empty_files}")
    print(f"  ERROR           : {color('ERROR',   str(result.error_count))}")
    print(f"  WARNING         : {color('WARNING', str(result.warning_count))}")
    print(f"  INFO            : {color('INFO',    str(result.info_count))}")

    # クラス分布
    if result.class_counts:
        print("\n  クラス分布:")
        total = sum(result.class_counts.values())
        for cls_name, cnt in sorted(result.class_counts.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            bar = "#" * int(pct / 2)
            print(f"    {cls_name:<20} {cnt:>6} ({pct:5.1f}%)  {bar}")

    # 問題一覧
    if result.issues:
        print("\n  検出した問題:")
        # ERRORを先に、次にWARNING、最後にINFO
        sorted_issues = sorted(result.issues, key=lambda i: {"ERROR": 0, "WARNING": 1, "INFO": 2}[i.level])
        for issue in sorted_issues:
            loc = f"{issue.file}" + (f":{issue.line}" if issue.line > 0 else "")
            tag = color(issue.level, f"[{issue.level:<7}]")
            print(f"  {tag} [{issue.code}] {loc}")
            print(f"           {issue.message}")
    else:
        print(f"\n  {color('INFO', '問題は検出されませんでした。')}")

    print("=" * 60)
    if result.error_count > 0:
        print(color("ERROR", f"  結果: FAILED ({result.error_count} ERROR) → 学習前に修正してください"))
    else:
        print(color("INFO", "  結果: PASSED"))
    print("=" * 60 + "\n")


def save_report_json(result: ValidationResult, out_path: Path) -> None:
    data = {
        "summary": {
            "total_files":   result.total_files,
            "total_objects": result.total_objects,
            "empty_files":   result.empty_files,
            "errors":        result.error_count,
            "warnings":      result.warning_count,
            "infos":         result.info_count,
            "passed":        result.error_count == 0,
        },
        "class_counts": result.class_counts,
        "issues": [asdict(i) for i in result.issues],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info(f"レポートを保存: {out_path}")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="YOLO形式アノテーションの品質チェックツール",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels_dir",   type=str, required=True,  help="YOLO labels ディレクトリ")
    parser.add_argument("--images_dir",   type=str, default=None,   help="画像ディレクトリ（孤立ファイル検出に使用）")
    parser.add_argument("--class_file",   type=str, default="classes.txt", help="classes.txt のパス")
    parser.add_argument("--iou_thresh",   type=float, default=0.95, help="重複ラベル検出の IoU 閾値")
    parser.add_argument("--min_size",     type=float, default=0.005,help="BBox最小辺長（正規化値）。これ未満を警告")
    parser.add_argument("--max_size",     type=float, default=0.99, help="BBox最大辺長（正規化値）。これ超えを警告")
    parser.add_argument("--min_samples",  type=int,   default=10,   help="クラスごとの最小サンプル数。未満を警告")
    parser.add_argument("--report_json",  type=str,   default=None, help="JSON レポートの出力先パス")
    parser.add_argument("--no_color",     action="store_true",      help="カラー出力を無効化（CI環境向け）")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir) if args.images_dir else None
    class_file = Path(args.class_file)

    if not labels_dir.exists():
        logger.error(f"labels ディレクトリが見つかりません: {labels_dir}")
        sys.exit(1)

    # classes.txt 読み込み
    class_list: list[str] = []
    if class_file.exists():
        class_list = [l.strip() for l in class_file.read_text().splitlines() if l.strip()]
        logger.info(f"classes.txt: {len(class_list)} クラス読み込み")
    else:
        logger.warning(f"classes.txt が見つかりません → クラスID範囲チェックをスキップ")

    # ラベルファイル列挙
    label_files = sorted([
        p for p in labels_dir.glob("*.txt")
        if p.parent == labels_dir and p.name != "classes.txt"
    ])

    if not label_files:
        logger.error(f"有効なラベルファイルが見つかりません: {labels_dir}")
        sys.exit(1)

    logger.info(f"{len(label_files)} 件のラベルファイルを検証します")

    result = ValidationResult()

    # ── ファイル単位チェック ──
    all_bboxes: dict[str, list[BBox]] = {}
    iterator = tqdm(label_files, desc="ファイル検証") if TQDM_AVAILABLE else label_files

    for label_path in iterator:
        bboxes = validate_file(label_path, class_list, args.min_size, args.max_size, result)
        all_bboxes[label_path.name] = bboxes

        # 重複ラベルチェック（同一ファイル内）
        if len(bboxes) >= 2:
            check_duplicate_labels(label_path.name, bboxes, args.iou_thresh, result)

    # ── データセット全体チェック ──
    if images_dir and images_dir.exists():
        check_orphan_files(labels_dir, images_dir, result)
    elif args.images_dir:
        logger.warning(f"images ディレクトリが見つかりません: {images_dir} → 孤立ファイルチェックをスキップ")

    check_class_imbalance(class_list, result.class_counts, args.min_samples, result)

    # ── レポート出力 ──
    print_report(result, class_list, use_color=not args.no_color)

    if args.report_json:
        save_report_json(result, Path(args.report_json))

    # exit code で CI/CD 連携
    sys.exit(1 if result.error_count > 0 else 0)


if __name__ == "__main__":
    main()
