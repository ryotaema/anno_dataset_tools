"""画像とラベルの名前を一括で変更する。

セッションが違っても `1color.jpg` のような同じ名前が使われていると、1つの
フォルダに集約した時点で衝突し、学習時にどのデータか追えなくなる。そこで
ファイル名そのものに「どのデータか」を書き込んでおく。

    260101_lab/color/1color.jpg  →  260101_lab/color/260101_lab_1color_c.jpg

名前の作り方は3通り:

    かんたん    先頭・末尾に文字列を足す（親ディレクトリ名を提案する）
    テンプレート {parent2}_{stem}_{mod} のようにトークンで組み立てる
    標準命名     {cam}_{YYMMDD}_{HHMMSS}_{NNNNN}_{mod} に変換する
                （セッション単位の分割や、カメラ・取得日ごとの集計が使えるようになる）

画像と同じ stem のラベル（labels/ の .txt、labelImg の .xml）も同じ名前に
追従させる。他の操作と同じく「計画を作る（build_plan）」と「実行する
（apply_plan）」に分けてあり、GUI では変更後の名前を確認してから実行できる。

実行内容は rename_log.json に記録し、いつでも元の名前に戻せる。
"""

import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import naming
from .config import CONFIG
from .dataset import IMAGE_EXTS

LABEL_EXTS = {'.txt', '.xml'}
LOG_NAME = 'rename_log.json'

# ディレクトリ名・ファイル名に含まれるモダリティ → 命名規則のコード
# （命名規則の {mod} に使うコード）
MOD_CODES = {
    'color':          'c',
    'depth':          'd',
    'depth_colormap': 'dc',
    'ir':             'i1',
    'ir_left':        'i1',
    'ir_right':       'i2',
    'ir_left_color':  'i1c',
    'ir_right_color': 'i2c',
    'detected':       'det',
    'annotated':      'det',
}
# 個人設定（config.local.json の mod_codes）で追加・上書きできる
MOD_CODES.update({k.lower(): v for k, v in CONFIG['mod_codes'].items()})

# ファイル名からモダリティを読むときの候補（長いものから順に判定する）
_MOD_WORDS = sorted(MOD_CODES, key=len, reverse=True)

# 変換済み（標準命名）のファイル。二重変換を避けるため除外できる
CONVERTED_RE = re.compile(r'^[A-Za-z0-9]+_\d{6}_\d{6}_\d{5}_[A-Za-z0-9]+$')

# テンプレートで使えるトークン（GUI のヘルプに出す）
TOKENS = [
    ('{stem}',    '元のファイル名（拡張子なし）'),
    ('{parent}',  '1つ上のフォルダ名'),
    ('{parent2}', '2つ上のフォルダ名'),
    ('{parent3}', '3つ上のフォルダ名'),
    ('{mod}',     'モダリティ（color→c / depth→d）'),
    ('{n}',       '連番（{n:05d} で 00001 のように桁を揃える）'),
    ('{cam}',     'カメラ（設定欄の値）'),
    ('{date}',    '取得日 YYMMDD（設定欄の値）'),
    ('{time}',    '時刻 HHMMSS（設定欄の値）'),
]


def natural_key(text):
    """'1color, 2color, 10color' の順に並ぶソートキー。"""
    return [(0, int(p), '') if p.isdigit() else (1, 0, p)
            for p in re.split(r'(\d+)', str(text))]


def split_mod_word(stem):
    """ファイル名からモダリティ語を切り出して (残り, モダリティコード) を返す。

    '1color' → ('1', 'c') / 'img_depth' → ('img_', 'd') / '1' → ('1', None)
    同じショットの color/depth に同じ連番を振るために使う。
    """
    low = stem.lower()
    for word in _MOD_WORDS:
        idx = low.find(word)
        if idx >= 0:
            return stem[:idx] + stem[idx + len(word):], MOD_CODES[word]
    return stem, None


# --------------------------------------------------------------- 対象を集める

@dataclass(eq=False)        # 連番の対応表に使うので、同じ内容でも別物として扱う
class Group:
    """1件（画像1枚と、それに追従するラベル）。"""
    stem:   str
    image:  Path = None
    labels: list = field(default_factory=list)

    @property
    def anchor(self):
        """名前の元にするパス（画像が無ければラベル）。"""
        return self.image or (self.labels[0] if self.labels else None)

    @property
    def files(self):
        return ([self.image] if self.image else []) + list(self.labels)


def _labels_for(image, include_labels, labels_dir=None):
    """画像に対応するラベル（同じ stem の .txt / .xml）を探す。

    見る場所は次の3つ。よくある置き方はこれで拾える。

        同じフォルダ                      labelImg の既定
        パス中の 'images' を 'labels' に   images/labels 形式・train/images 形式
        1〜2つ上のフォルダの labels/       260101_lab/color + 260101_lab/labels
    """
    if not include_labels:
        return []
    found = []
    dirs = [image.parent]
    parts = list(image.parent.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == 'images':
            dirs.append(Path(*parts[:i], 'labels', *parts[i + 1:]))
            break
    dirs.append(image.parent / 'labels')
    dirs.append(image.parent.parent / 'labels')
    if labels_dir:
        dirs.append(Path(labels_dir))

    for d in dirs:
        for ext in sorted(LABEL_EXTS):
            p = d / (image.stem + ext)
            if p.is_file() and p not in found:
                found.append(p)
    return found


def collect(root, recursive=True, include_labels=True, labels_dir=None,
            skip_converted=False):
    """対象を集める。

    Returns:
        (groups, notes)  notes は補足メッセージのリスト
    """
    root = Path(root).expanduser().resolve()
    notes = []
    it = root.rglob('*') if recursive else root.glob('*')
    files = sorted((p for p in it if p.is_file()), key=lambda p: natural_key(str(p)))

    groups, taken = [], set()
    for p in files:
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if skip_converted and CONVERTED_RE.match(p.stem):
            continue
        labels = _labels_for(p, include_labels, labels_dir)
        taken.update(labels)
        groups.append(Group(stem=p.stem, image=p, labels=labels))

    # 画像のないラベル（labels だけのフォルダを選んだ場合も含む）
    orphans = [p for p in files
               if p.suffix.lower() in LABEL_EXTS and p not in taken
               and p.name != 'classes.txt'
               and not (skip_converted and CONVERTED_RE.match(p.stem))]
    if orphans and include_labels:
        for p in orphans:
            groups.append(Group(stem=p.stem, image=None, labels=[p]))
        notes.append(f'画像のないラベルが {len(orphans)} 件あります（同じ規則で変更します）')

    if not groups:
        notes.append('対象のファイルが見つかりませんでした')
    return groups, notes


# --------------------------------------------------------------- 名前を作る

@dataclass
class Spec:
    """名前の付け方。"""
    mode:     str = 'affix'      # 'affix' / 'template' / 'standard'
    prefix:   str = ''
    suffix:   str = ''
    template: str = '{parent2}_{stem}_{mod}'
    cam:      str = ''
    date:     str = ''           # YYMMDD
    time:     str = ''           # HHMMSS
    lower:    bool = False       # 拡張子を小文字にそろえる
    dedupe:   bool = True        # 名前が重なったら連番を足して回避する

    def effective_template(self):
        if self.mode == 'standard':
            return '{cam}_{date}_{time}_{n:05d}_{mod}'
        if self.mode == 'template':
            return self.template
        return self.prefix + '{stem}' + self.suffix


class _Tokens(dict):
    """未定義のトークンは空文字にする（テンプレートの書き間違いで落ちないように）。"""

    def __missing__(self, key):
        return ''


def tokens_for(group, spec, index):
    """テンプレートに渡す値を作る。"""
    path = group.anchor
    parts = path.parent.parts
    mod = ''
    if parts:
        mod = MOD_CODES.get(parts[-1].lower(), '')
    if not mod:
        mod = split_mod_word(path.stem)[1] or ''
    if not mod:
        info = naming.parse(path.stem)
        mod = info.mod or ''

    return _Tokens(
        stem=path.stem,
        parent=parts[-1] if len(parts) >= 1 else '',
        parent2=parts[-2] if len(parts) >= 2 else '',
        parent3=parts[-3] if len(parts) >= 3 else '',
        dir=parts[-1] if len(parts) >= 1 else '',
        mod=mod,
        n=index, index=index,
        cam=spec.cam, date=spec.date, time=spec.time,
    )


def _fmt(template, tokens):
    try:
        return template.format_map(tokens)
    except (ValueError, KeyError, IndexError) as e:
        raise ValueError(f'テンプレートが正しくありません: {template}（{e}）')


def new_stem(group, spec, index):
    """1件分の新しい stem を作る。"""
    tokens = tokens_for(group, spec, index)

    if spec.mode == 'affix':
        # すでに付いていれば足さない（同じ設定で2回実行しても二重にならない）
        prefix, suffix = _fmt(spec.prefix, tokens), _fmt(spec.suffix, tokens)
        stem = tokens['stem']
        if prefix and not stem.startswith(prefix):
            stem = prefix + stem
        if suffix and not stem.endswith(suffix):
            stem = stem + suffix
    else:
        stem = _fmt(spec.effective_template(), tokens)

    stem = re.sub(r'[/\\]', '_', stem).strip()
    stem = re.sub(r'_{3,}', '__', stem)
    return stem


def shot_numbers(groups):
    """同じショット（1color / 1depth）に同じ連番を振る。"""
    keys = {}
    for g in groups:
        keys[g] = split_mod_word(g.anchor.stem)[0] if g.anchor else g.stem
    order = sorted(set(keys.values()), key=natural_key)
    number = {k: i for i, k in enumerate(order, start=1)}
    return {g: number[k] for g, k in keys.items()}


# --------------------------------------------------------------- 変更計画

@dataclass
class Plan:
    root:      Path
    spec:      Spec
    mode:      str = 'inplace'          # 'inplace' / 'copy'
    out_dir:   Path = None
    pairs:     list = field(default_factory=list)   # (src, dst, 種別)
    rel_base:  Path = None              # コピー出力でフォルダ構成を保つための基準
    unchanged: int = 0
    conflicts: list = field(default_factory=list)   # (dst, [src, ...])
    notes:     list = field(default_factory=list)

    @property
    def n(self):
        return len(self.pairs)

    @property
    def n_images(self):
        return sum(1 for _, _, kind in self.pairs if kind == '画像')

    @property
    def n_labels(self):
        return sum(1 for _, _, kind in self.pairs if kind == 'ラベル')

    @property
    def ok(self):
        return bool(self.pairs) and not self.conflicts


def build_plan(root, spec, recursive=True, include_labels=True, labels_dir=None,
               skip_converted=False, mode='inplace', out_dir=None):
    """変更計画を作る。ここではファイルに触らない。"""
    root = Path(root).expanduser().resolve()
    groups, notes = collect(root, recursive, include_labels, labels_dir, skip_converted)
    plan = Plan(root=root, spec=spec, mode=mode,
                out_dir=Path(out_dir).expanduser().resolve() if out_dir else None,
                notes=list(notes))

    # コピー出力では元のフォルダ構成をそのまま写す。対象より外にあるラベル
    # （260101_lab/color を選んだときの 260101_lab/labels など）も
    # 位置関係を保てるように、共通の親を基準にする。
    dirs = [str(root)] + [str(f.parent) for g in groups for f in g.files]
    plan.rel_base = Path(os.path.commonpath(dirs)) if dirs else root

    numbers = shot_numbers(groups)
    used = defaultdict(list)        # 出力先 → 元ファイル（衝突の検出用）
    claimed = set()                 # 既に確保した出力先

    for g in sorted(groups, key=lambda g: natural_key(str(g.anchor))):
        stem = new_stem(g, spec, numbers[g])
        if not stem:
            plan.notes.append(f'新しい名前が空になりました: {g.anchor}')
            continue

        # 同じ名前が既に出ていたら連番を足す（画像とラベルは同じ名前のまま揃える）
        if spec.dedupe:
            i = 2
            while any(_dst_of(f, stem, plan) in claimed for f in g.files):
                stem = f'{new_stem(g, spec, numbers[g])}_{i}'
                i += 1

        for f in g.files:
            dst = _dst_of(f, stem, plan)
            kind = '画像' if f.suffix.lower() in IMAGE_EXTS else 'ラベル'
            claimed.add(dst)
            used[dst].append(f)
            if dst == f:
                plan.unchanged += 1
                continue
            plan.pairs.append((f, dst, kind))

    # 衝突: 同じ出力先に2つ以上／既にあるファイルとぶつかる
    srcs = {src for src, _, _ in plan.pairs}
    for dst, sources in used.items():
        if len(sources) > 1:
            plan.conflicts.append((dst, sources))
        elif dst.exists() and dst not in srcs and dst != sources[0]:
            plan.conflicts.append((dst, sources))
    return plan


def _dst_of(src, stem, plan):
    """1ファイルの出力先。"""
    ext = src.suffix.lower() if plan.spec.lower else src.suffix
    if plan.mode == 'copy' and plan.out_dir:
        try:
            rel = src.parent.relative_to(plan.rel_base or plan.root)
        except ValueError:
            rel = Path(src.parent.name)
        return plan.out_dir / rel / (stem + ext)
    return src.parent / (stem + ext)


# --------------------------------------------------------------- 実行

def apply_plan(plan, on_progress=None):
    """計画を実行する。

    その場でリネームする場合は一時名を経由した2段階で行う。A→B・B→A のような
    入れ替えでも途中で上書きが起きないようにするため。

    Returns:
        {'renamed': n, 'errors': [(src, msg)], 'log': ログのパス}
    """
    stats = {'renamed': 0, 'errors': [], 'log': None}
    total = plan.n
    if not total:
        return stats

    if plan.mode == 'copy':
        for i, (src, dst, _) in enumerate(plan.pairs, 1):
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                stats['renamed'] += 1
            except OSError as e:
                stats['errors'].append((src, str(e)))
            if on_progress and (i % 50 == 0 or i == total):
                on_progress(i, total)
    else:
        stamp = datetime.now().strftime('%H%M%S%f')
        temps = []
        for i, (src, dst, _) in enumerate(plan.pairs, 1):
            tmp = src.with_name(f'.rntmp{stamp}_{i}{src.suffix}')
            try:
                src.rename(tmp)
                temps.append((tmp, dst, src))
            except OSError as e:
                stats['errors'].append((src, str(e)))
            if on_progress and (i % 50 == 0 or i == total):
                on_progress(i, total * 2)

        for i, (tmp, dst, src) in enumerate(temps, 1):
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp.rename(dst)
                stats['renamed'] += 1
            except OSError as e:
                stats['errors'].append((src, str(e)))
                try:
                    tmp.rename(src)          # 戻せるものは元に戻す
                except OSError:
                    pass
            if on_progress and (i % 50 == 0 or i == total):
                on_progress(total + i, total * 2)

    stats['log'] = write_log(plan, stats)
    return stats


def log_path(plan):
    base = plan.out_dir if (plan.mode == 'copy' and plan.out_dir) else plan.root
    return base / LOG_NAME


def write_log(plan, stats):
    """実行内容を rename_log.json に追記する（元に戻すために使う）。"""
    path = log_path(plan)
    data = {'runs': []}
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            data.setdefault('runs', [])
        except (OSError, json.JSONDecodeError):
            data = {'runs': []}

    data['runs'].append({
        'at':    datetime.now().isoformat(timespec='seconds'),
        'root':  str(plan.root),
        'mode':  plan.mode,
        'renamed': stats['renamed'],
        'pairs': [[str(src), str(dst)] for src, dst, _ in plan.pairs],
    })
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError:
        return None
    return path


def read_log(path):
    """rename_log.json を読む。無ければ None。"""
    path = Path(path)
    if path.is_dir():
        path = path / LOG_NAME
    if not path.is_file():
        return None, None
    try:
        with open(path) as f:
            return json.load(f), path
    except (OSError, json.JSONDecodeError):
        return None, path


def undo_last(root, on_progress=None):
    """最後に行った変更を取り消して、元の名前に戻す。

    Returns:
        (戻した件数, メッセージ)
    """
    data, path = read_log(root)
    if not data or not data.get('runs'):
        return 0, '取り消せる履歴がありません（rename_log.json が見つかりません）'

    run = data['runs'][-1]
    if run.get('mode') == 'copy':
        return 0, ('最後の実行はコピー出力のため取り消しできません。'
                   f'出力先フォルダを削除してください: {run.get("root")}')

    pairs = [(Path(dst), Path(src)) for src, dst in run['pairs']]
    total = len(pairs)
    stamp = datetime.now().strftime('%H%M%S%f')
    temps, errors = [], []

    for i, (src, dst) in enumerate(pairs, 1):
        if not src.exists():
            errors.append(f'見つかりません: {src}')
            continue
        tmp = src.with_name(f'.rntmp{stamp}_{i}{src.suffix}')
        try:
            src.rename(tmp)
            temps.append((tmp, dst))
        except OSError as e:
            errors.append(f'{src}: {e}')
        if on_progress and (i % 50 == 0 or i == total):
            on_progress(i, total * 2)

    done = 0
    for i, (tmp, dst) in enumerate(temps, 1):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp.rename(dst)
            done += 1
        except OSError as e:
            errors.append(f'{dst}: {e}')
        if on_progress and (i % 50 == 0 or i == total):
            on_progress(total + i, total * 2)

    data['runs'].pop()
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError:
        pass

    msg = f'{done:,} 件を元の名前に戻しました'
    if errors:
        msg += f'（{len(errors)} 件は戻せませんでした）'
    return done, msg


# --------------------------------------------------------------- 提案

def suggest(root):
    """選んだフォルダから、名前の付け方の候補を作る。

    260101_lab/color を選んだ場合:
        prefix='260101_lab_' / suffix='_c' / template='{parent2}_{stem}_{mod}'

    Returns:
        {'prefix': str, 'suffix': str, 'template': str, 'why': 説明}
    """
    root = Path(root).expanduser().resolve()
    name = root.name
    low = name.lower()

    if low in MOD_CODES:
        return {'prefix': f'{root.parent.name}_', 'suffix': f'_{MOD_CODES[low]}',
                'template': '{parent2}_{stem}_{mod}',
                'why': f'モダリティ別フォルダなので、親フォルダ名「{root.parent.name}」を'
                       f'先頭に、モダリティ「{MOD_CODES[low]}」を末尾に付けます'}

    if low in ('images', 'labels', 'train', 'val', 'valid', 'test'):
        return {'prefix': f'{root.parent.name}_', 'suffix': '',
                'template': '{parent2}_{stem}',
                'why': f'データセットのフォルダなので、親フォルダ名「{root.parent.name}」を'
                       '先頭に付けます'}

    # 配下がモダリティ別に分かれているなら、モダリティは {mod} で自動的に付ける
    subs = [d.name.lower() for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
    if any(s in MOD_CODES for s in subs):
        return {'prefix': f'{name}_', 'suffix': '_{mod}',
                'template': '{parent2}_{stem}_{mod}',
                'why': f'配下がモダリティ別なので、「{name}」を先頭に、'
                       'モダリティ（color→c など）を末尾に付けます'}

    return {'prefix': f'{name}_', 'suffix': '',
            'template': '{parent}_{stem}',
            'why': f'このフォルダ名「{name}」を先頭に付けます'}
