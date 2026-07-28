"""データセットの整理操作。

    collect  複数のデータセットを1つにまとめる（アノテーション用・zip出力）
    split    train / val / test に分ける（割合でも枚数でも指定できる）
    subset   条件で絞り込み、枚数や割合で間引く

いずれも「計画を作る（plan_*）」と「書き出す（write_plan）」を分けている。
GUI では計画を表で確認してから実行できる。
"""

import random
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .dataset import class_name

# 出力レイアウト
LAYOUTS = {
    'flat':  'images/ labels/（分割なし・アノテーション用）',
    'split': 'images/train/ labels/train/（split_dataset.py と同じ形）',
    'yolo':  'train/images/ train/labels/（YOLO 標準・data.yaml 付き）',
}

# 分割時にひとまとまりとして扱う単位
GROUP_MODES = {
    'none':    '画像ごと（従来どおり）',
    'shot':    '同じショットは同じ split へ（color/depth などを分離しない）',
    'session': '同じセッションは同じ split へ（似た画像による情報漏れを防ぐ）',
}


@dataclass
class Plan:
    """出力計画。entries は (Item, split, 出力stem) の並び。"""
    entries: list = field(default_factory=list)
    out_dir: Path = None
    layout:  str = 'flat'
    renamed: int = 0        # 名前衝突でリネームした数
    skipped: list = field(default_factory=list)   # (Item, 理由)

    @property
    def n(self):
        return len(self.entries)

    def counts_by_split(self):
        c = Counter()
        for _, split, _ in self.entries:
            c[split or '(なし)'] += 1
        return c

    def items_of(self, split):
        return [it for it, s, _ in self.entries if s == split]


# ------------------------------------------------------------ グループ化

def group_key(item, mode):
    """分割時に同じ組として扱うキー。"""
    if mode == 'session':
        return item.info.session or f'__{item.source}__{item.stem}'
    if mode == 'shot':
        if item.info.session and item.info.shot is not None:
            return f'{item.info.session}_{item.info.shot:05d}'
        return f'__{item.source}__{item.stem}'
    return f'__{item.source}__{item.split}__{item.stem}'


def primary_class(item):
    """stratified 分割用の代表クラス。ラベルなしは None。"""
    return item.class_ids[0] if item.class_ids else None


# ------------------------------------------------------------ 分割

def split_items(items, mode='ratio', ratios=(0.8, 0.2, 0.0), counts=(0, 0, 0),
                stratified=False, group='none', seed=42, names=('train', 'val', 'test')):
    """train/val/test に振り分ける。

    Args:
        mode:  'ratio' なら ratios（合計1.0）、'count' なら counts（絶対枚数）で決める。
        group: 'none' / 'shot' / 'session'（GROUP_MODES）
    Returns:
        {split名: [Item, ...]}  余りは最初の split に入る。
    """
    rng = random.Random(seed)

    # グループ単位にまとめる（group='none' なら1件1グループ）
    groups = defaultdict(list)
    for item in items:
        groups[group_key(item, group)].append(item)
    keys = list(groups)

    # stratified: グループの代表クラスごとに分けてから同じ比率で配る
    if stratified:
        buckets = defaultdict(list)
        for k in keys:
            cls = next((primary_class(i) for i in groups[k]
                        if primary_class(i) is not None), None)
            buckets[cls].append(k)
    else:
        buckets = {None: keys}

    total_items = len(items)
    result = {n: [] for n in names[:3]}

    for _, bucket_keys in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        rng.shuffle(bucket_keys)
        bucket_items = sum(len(groups[k]) for k in bucket_keys)

        # 目標は「グループ数」ではなく「画像枚数」で持つ。
        # グループ単位で分けるときもサイズがまちまちなので、
        # 残り必要枚数が最も多い split へ順に入れて比率を保つ。
        if mode == 'count':
            share = (bucket_items / total_items) if total_items else 0
            need = [c * share for c in counts[:3]]
        else:
            need = [bucket_items * r for r in ratios[:3]]

        for gk in bucket_keys:
            size = len(groups[gk])
            best = max(range(3), key=lambda i: need[i])
            if need[best] <= 0:
                if mode == 'count':
                    break          # 指定枚数に達したので残りは使わない（＝間引き）
                best = 0           # 割合指定の端数は train に寄せる
            result[names[best]].extend(groups[gk])
            need[best] -= size

    return {k: v for k, v in result.items() if v or k in names[:2]}


# ------------------------------------------------------------ 絞り込み・間引き

def filter_items(items, filters, classes=None):
    """filters = {キー: 許可する値の集合} で絞り込む（空の集合は無条件）。"""
    out = []
    for item in items:
        ok = True
        for key, allowed in filters.items():
            if not allowed:
                continue
            values = item.value_list(key)
            if key == 'class' and classes:
                values = [class_name(classes, v) if v != '(ラベルなし)' else v
                          for v in values]
            if not any(v in allowed for v in values):
                ok = False
                break
        if ok:
            out.append(item)
    return out


def limit_items(items, mode='none', value=0, per_class=False, classes=None,
                strategy='random', group='none', seed=42):
    """枚数・割合の上限で間引く。

    Args:
        mode:     'none' / 'count'（N枚まで） / 'ratio'（全体のP%まで）
        per_class: True ならクラスごとに上限を適用する（クラス不均衡の是正）
        strategy: 'random' / 'head'（先頭から） / 'spread'（セッションから均等に）
    """
    if mode == 'none' or not items:
        return list(items)

    def take(subset, limit):
        if limit >= len(subset):
            return list(subset)
        if strategy == 'head':
            return list(subset)[:limit]
        if strategy in ('spread', 'spread_random'):
            return _take_spread(subset, limit,
                                group if group != 'none' else 'session',
                                inner='random' if strategy == 'spread_random' else 'even',
                                seed=seed)
        picked = list(subset)
        random.Random(seed).shuffle(picked)
        return picked[:limit]

    def limit_for(n_total):
        if mode == 'count':
            return int(value)
        return max(0, int(round(n_total * float(value) / 100.0)))

    if not per_class:
        return take(items, limit_for(len(items)))

    # クラスごとに上限をかける（複数クラスを持つ画像はどれか1つに数える）
    by_class = defaultdict(list)
    for item in items:
        key = primary_class(item)
        by_class[key].append(item)

    kept = []
    for _, group_items in sorted(by_class.items(), key=lambda kv: str(kv[0])):
        kept.extend(take(group_items, limit_for(len(group_items))))
    return kept


def spreadable(items, group='session'):
    """spread が意味を持つデータか（グループが2つ以上あり、複数件を含むグループがあるか）。

    命名規則の対象外データはセッションが取れず1件1グループになるため、
    その場合 spread はランダム選択にフォールバックする。
    """
    counts = defaultdict(int)
    for item in items:
        counts[group_key(item, group)] += 1
    return len(counts) > 1 and max(counts.values()) > 1


def _take_spread(items, limit, group_mode, inner='even', seed=42):
    """グループ（既定はセッション）から均等に取り出す。

    Args:
        inner: グループ内での選び方。
               'even'   … 等間隔（撮影時刻がまんべんなく散る）
               'random' … ランダム（グループごとに固定 seed）
    """
    groups = defaultdict(list)
    for item in items:
        groups[group_key(item, group_mode)].append(item)
    order = sorted(groups)

    # グループ分けができない（1件1グループ）ならランダムに任せる。
    # 先頭から詰めると偏りが大きく、head と変わらなくなるため。
    if len(order) <= 1 or max(len(v) for v in groups.values()) <= 1:
        picked = list(items)
        random.Random(seed).shuffle(picked)
        return picked[:limit]

    # 1) 各グループの取得枚数を均等に配る（小さいグループは持っている分だけ）
    quota = {k: 0 for k in order}
    remaining = limit
    while remaining > 0:
        progressed = False
        for k in order:
            if quota[k] < len(groups[k]):
                quota[k] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    # 2) 各グループの中から実際に選ぶ
    picked = []
    for k in order:
        n, seq = quota[k], groups[k]
        if n <= 0:
            continue
        if n >= len(seq):
            picked.extend(seq)
        elif inner == 'random':
            rng = random.Random(f'{seed}:{k}')     # 文字列 seed なので再現性がある
            idxs = sorted(rng.sample(range(len(seq)), n))
            picked.extend(seq[i] for i in idxs)
        else:
            step = len(seq) / n
            picked.extend(seq[min(int(i * step), len(seq) - 1)] for i in range(n))
    return picked


# ------------------------------------------------------------ 出力計画

def plan_output(split_map, out_dir, layout='flat', prefix_source=False):
    """出力先の計画を作る。ファイル名の衝突はここで解決する。

    Args:
        split_map: {split名: [Item]}。分割しない場合は {'': [Item]}。
        prefix_source: True なら常に「取得元フォルダ名__」を付ける。
                       False でも衝突したときだけ付ける。
    """
    plan = Plan(out_dir=Path(out_dir), layout=layout)
    used = defaultdict(set)      # split → 使用済み stem

    for split, items in split_map.items():
        for item in items:
            if not item.has_image:
                plan.skipped.append((item, '画像がありません'))
                continue

            stem = f'{item.source}__{item.stem}' if prefix_source else item.stem
            if stem in used[split]:
                # 同名が既にある → 取得元名、それでも駄目なら連番を足す
                cand = f'{item.source}__{item.stem}'
                if cand in used[split]:
                    i = 2
                    while f'{cand}__{i}' in used[split]:
                        i += 1
                    cand = f'{cand}__{i}'
                stem = cand
                plan.renamed += 1

            used[split].add(stem)
            plan.entries.append((item, split, stem))
    return plan


def dest_paths(plan, item, split, stem):
    """1件の出力先 (画像パス, ラベルパス) を返す。"""
    out = plan.out_dir
    img_ext = item.image.suffix.lower() if item.has_image else '.jpg'
    if plan.layout == 'yolo':
        base = out / (split or 'train')
        return base / 'images' / (stem + img_ext), base / 'labels' / (stem + '.txt')
    if plan.layout == 'split':
        return (out / 'images' / split / (stem + img_ext),
                out / 'labels' / split / (stem + '.txt'))
    return out / 'images' / (stem + img_ext), out / 'labels' / (stem + '.txt')


def write_plan(plan, classes=None, move=False, on_progress=None):
    """計画を実行してファイルを書き出す。

    Returns:
        {'images': n, 'labels': n, 'errors': [(src, msg)]}
    """
    stats = {'images': 0, 'labels': 0, 'errors': []}
    total = plan.n

    for n, (item, split, stem) in enumerate(plan.entries, 1):
        img_dst, lbl_dst = dest_paths(plan, item, split, stem)
        try:
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(item.image), img_dst)
            else:
                shutil.copy2(item.image, img_dst)
            stats['images'] += 1

            if item.has_label:
                lbl_dst.parent.mkdir(parents=True, exist_ok=True)
                if move:
                    shutil.move(str(item.label), lbl_dst)
                else:
                    shutil.copy2(item.label, lbl_dst)
                stats['labels'] += 1
        except OSError as e:
            stats['errors'].append((item.image, str(e)))

        if on_progress and (n % 50 == 0 or n == total):
            on_progress(n, total)

    if classes:
        write_class_files(plan, classes)
    return stats


def write_class_files(plan, classes):
    """classes.txt（と yolo レイアウトなら data.yaml）を書く。"""
    out = plan.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'classes.txt').write_text('\n'.join(classes) + '\n')

    if plan.layout != 'yolo':
        return
    splits = sorted({s for _, s, _ in plan.entries if s})
    lines = [f'path: {out}']
    for s in splits:
        key = 'val' if s in ('val', 'valid') else s
        lines.append(f'{key}: {s}/images')
    lines.append(f'nc: {len(classes)}')
    lines.append('names:')
    lines.extend(f'  {i}: {name}' for i, name in enumerate(classes))
    (out / 'data.yaml').write_text('\n'.join(lines) + '\n')


def make_zip(src_dir, zip_path, on_progress=None):
    """出力フォルダを zip にまとめる（CVAT へのアップロード等に使う）。"""
    src_dir, zip_path = Path(src_dir), Path(zip_path)
    files = [p for p in sorted(src_dir.rglob('*')) if p.is_file()]
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for n, p in enumerate(files, 1):
            zf.write(p, p.relative_to(src_dir))
            if on_progress and (n % 50 == 0 or n == len(files)):
                on_progress(n, len(files))
    return len(files), zip_path.stat().st_size
