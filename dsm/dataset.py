"""データセットの走査と集計。

対応レイアウト（自動判別）:

    A. images/labels フラット      <root>/images/*.jpg      <root>/labels/*.txt
    B. images/labels + split      <root>/images/train/*.jpg <root>/labels/train/*.txt
    C. split + images/labels      <root>/train/images/*.jpg <root>/train/labels/*.txt
    D. 同一ディレクトリ混在        <root>/*.jpg  <root>/*.txt   （labelImg の既定）

ラベルが1つも無いディレクトリ（アノテーション前の画像だけ）も読み込める。
"""

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import naming

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
SPLIT_NAMES = ('train', 'val', 'valid', 'test')

# 集計・絞り込みに使えるキー → 表示名
GROUP_KEYS = {
    'class':   'クラス',
    'cam':     'カメラ',
    'date':    '取得日',
    'session': 'セッション',
    'split':   'split',
    'source':  '取得元データセット',
    'mod':     'モダリティ',
}


@dataclass
class Item:
    """画像1枚とそのラベル。"""
    stem:      str
    image:     Path = None
    label:     Path = None
    split:     str = ''
    root:      Path = None          # 由来のデータセットルート
    info:      naming.NameInfo = naming.NO_INFO
    class_ids: tuple = ()           # ラベルに含まれるクラスID（重複なし）
    n_boxes:   int = 0

    @property
    def has_image(self):
        return self.image is not None

    @property
    def has_label(self):
        return self.label is not None

    @property
    def source(self):
        return self.root.name if self.root else naming.UNKNOWN

    def value(self, key):
        """集計・絞り込み用の値。'class' だけは複数値なので value_list を使うこと。"""
        if key == 'split':
            return self.split or '(未分割)'
        if key == 'source':
            return self.source
        return self.info.value(key)

    def value_list(self, key):
        """1件が複数の値を持ちうるキー（class）に対応した取得。"""
        if key == 'class':
            return [str(c) for c in self.class_ids] or ['(ラベルなし)']
        return [self.value(key)]


@dataclass
class Dataset:
    """1つのデータセットルート。"""
    root:    Path
    items:   list = field(default_factory=list)
    classes: list = field(default_factory=list)   # クラス名（index = クラスID）
    layout:  str = ''
    warnings: list = field(default_factory=list)

    @property
    def n_images(self):
        return sum(1 for i in self.items if i.has_image)

    @property
    def n_labeled(self):
        return sum(1 for i in self.items if i.has_label)


# --------------------------------------------------------------- 走査

def _iter_images(d):
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def detect_layout(root):
    """(レイアウト記号, [(images_dir, labels_dir, split), ...]) を返す。"""
    root = Path(root)
    img_root, lbl_root = root / 'images', root / 'labels'

    # B: images/train, labels/train
    if img_root.is_dir():
        splits = [d.name for d in sorted(img_root.iterdir())
                  if d.is_dir() and d.name in SPLIT_NAMES]
        if splits:
            return 'B', [(img_root / s, lbl_root / s, s) for s in splits]
        return 'A', [(img_root, lbl_root if lbl_root.is_dir() else None, '')]

    # C: train/images, train/labels
    pairs = [(root / s / 'images', root / s / 'labels', s)
             for s in SPLIT_NAMES if (root / s / 'images').is_dir()]
    if pairs:
        return 'C', [(i, l if l.is_dir() else None, s) for i, l, s in pairs]

    # D: 同一ディレクトリに画像とラベルが混在
    if root.is_dir() and any(p.suffix.lower() in IMAGE_EXTS
                             for p in root.iterdir() if p.is_file()):
        return 'D', [(root, root, '')]

    return '', []


def load_classes(root):
    """classes.txt / data.yaml からクラス名リストを読む。無ければ空リスト。"""
    root = Path(root)
    for name in ('classes.txt', 'labels/classes.txt', 'obj.names'):
        p = root / name
        if p.is_file():
            names = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
            if names:
                return names

    for name in ('data.yaml', 'dataset.yaml', 'data.yml'):
        p = root / name
        if not p.is_file():
            continue
        try:
            import yaml
            data = yaml.safe_load(p.read_text())
        except Exception:
            continue
        names = (data or {}).get('names')
        if isinstance(names, dict):          # {0: 'pepper', 1: ...}
            return [names[k] for k in sorted(names)]
        if isinstance(names, list):
            return list(names)
    return []


def _read_label(path):
    """YOLO ラベルを読んで (クラスIDのtuple, BBox数) を返す。"""
    ids, n = set(), 0
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                ids.add(int(float(parts[0])))
            except ValueError:
                continue
            n += 1
    except OSError:
        pass
    return tuple(sorted(ids)), n


def load(root, read_labels=True, progress=None):
    """データセットを読み込む。

    Args:
        read_labels: True ならラベルを開いてクラス構成まで集計する。
        progress:    callback(done, total) 進捗通知（任意）。
    """
    root = Path(root).expanduser().resolve()
    ds = Dataset(root=root, classes=load_classes(root))
    ds.layout, pairs = detect_layout(root)
    if not pairs:
        ds.warnings.append('画像が見つかりませんでした（images/ か画像ファイルを含むフォルダを選んでください）')
        return ds

    # 総数を数えてから読む（進捗表示のため）
    all_images = []
    for img_dir, lbl_dir, split in pairs:
        if img_dir.is_dir():
            for p in _iter_images(img_dir):
                all_images.append((p, lbl_dir, split))
    total = len(all_images)

    labeled_stems = set()
    for n, (img, lbl_dir, split) in enumerate(all_images, 1):
        label = None
        if lbl_dir is not None:
            cand = lbl_dir / (img.stem + '.txt')
            if cand.is_file():
                label = cand
        class_ids, n_boxes = ((), 0)
        if label is not None and read_labels:
            class_ids, n_boxes = _read_label(label)
        if label is not None:
            labeled_stems.add((split, img.stem))
        ds.items.append(Item(stem=img.stem, image=img, label=label, split=split,
                             root=root, info=naming.parse(img.stem),
                             class_ids=class_ids, n_boxes=n_boxes))
        if progress and (n % 200 == 0 or n == total):
            progress(n, total)

    # 画像のない孤立ラベルも拾う（分割漏れの発見用）
    for img_dir, lbl_dir, split in pairs:
        if lbl_dir is None or not lbl_dir.is_dir():
            continue
        for lp in sorted(lbl_dir.glob('*.txt')):
            if lp.name == 'classes.txt' or (split, lp.stem) in labeled_stems:
                continue
            class_ids, n_boxes = _read_label(lp) if read_labels else ((), 0)
            ds.items.append(Item(stem=lp.stem, image=None, label=lp, split=split,
                                 root=root, info=naming.parse(lp.stem),
                                 class_ids=class_ids, n_boxes=n_boxes))

    n_orphan = sum(1 for i in ds.items if not i.has_image)
    if n_orphan:
        ds.warnings.append(f'画像のないラベルが {n_orphan} 件あります')
    n_unlabeled = sum(1 for i in ds.items if i.has_image and not i.has_label)
    if n_unlabeled:
        ds.warnings.append(f'ラベルのない画像が {n_unlabeled} 件あります')
    return ds


# --------------------------------------------------------------- 集計

def class_name(classes, cid):
    try:
        return f'{cid}: {classes[int(cid)]}'
    except (ValueError, IndexError, TypeError):
        return str(cid)


def counts_by(items, key, classes=None):
    """キー別の枚数を数える。

    Returns:
        Counter  {表示名: 枚数}
        'class' の場合、1枚に複数クラスがあれば各クラスに1ずつ加算する
        （＝合計は総枚数と一致しないことがある）。
    """
    c = Counter()
    for item in items:
        for v in item.value_list(key):
            if key == 'class' and classes and v not in ('(ラベルなし)',):
                v = class_name(classes, v)
            c[v] += 1
    return c


def box_counts_by_class(items, classes=None):
    """クラス別の BBox 数（インスタンス数）。"""
    c = Counter()
    for item in items:
        if not item.has_label:
            continue
        try:
            lines = item.label.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            c[class_name(classes, cid) if classes else str(cid)] += 1
    return c


def summary_rows(items, key, classes=None, total=None):
    """(値, 枚数, 割合) の行を枚数降順で返す。GUI の表にそのまま流し込む。"""
    counts = counts_by(items, key, classes)
    total = total if total is not None else len(items)
    rows = []
    for value, n in counts.most_common():
        ratio = (n / total * 100) if total else 0.0
        rows.append((value, n, ratio))
    return rows


def distinct_values(items, key, classes=None):
    """絞り込み用の選択肢（枚数の多い順）。"""
    return [v for v, _ in counts_by(items, key, classes).most_common()]


def write_summary_csv(path, items, classes, keys=None):
    """内訳を CSV に書き出す。"""
    keys = keys or ['class', 'cam', 'date', 'session', 'split']
    total = len(items)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['区分', '値', '枚数', '割合(%)'])
        for key in keys:
            for value, n, ratio in summary_rows(items, key, classes, total):
                w.writerow([GROUP_KEYS.get(key, key), value, n, f'{ratio:.1f}'])
